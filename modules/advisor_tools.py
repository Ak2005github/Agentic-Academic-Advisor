import sqlite3
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "academic.db"


def normalize_course_code(course_code: str) -> str:
    return course_code.strip().upper().replace(" ", "")


class AcademicAdvisorTools:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def get_course_info(self, course_code: str) -> Dict:
        course_code = normalize_course_code(course_code)

        with self._connect() as connection:
            course = connection.execute(
                """
                SELECT course_code, course_name, credits, department, description
                FROM courses
                WHERE course_code = ?
                """,
                (course_code,),
            ).fetchone()

            if course is None:
                return {
                    "found": False,
                    "course_code": course_code,
                    "message": f"No course found for {course_code}.",
                }

            prerequisites = connection.execute(
                """
                SELECT p.prerequisite_code, c.course_name
                FROM prerequisites p
                JOIN courses c ON c.course_code = p.prerequisite_code
                WHERE p.course_code = ?
                ORDER BY p.prerequisite_code
                """,
                (course_code,),
            ).fetchall()

            required_by = connection.execute(
                """
                SELECT p.course_code, c.course_name
                FROM prerequisites p
                JOIN courses c ON c.course_code = p.course_code
                WHERE p.prerequisite_code = ?
                ORDER BY p.course_code
                """,
                (course_code,),
            ).fetchall()

        return {
            "found": True,
            "course_code": course["course_code"],
            "course_name": course["course_name"],
            "credits": course["credits"],
            "department": course["department"],
            "description": course["description"],
            "prerequisites": [dict(row) for row in prerequisites],
            "required_by": [dict(row) for row in required_by],
        }

    def search_courses(self, query: str, limit: int = 8) -> List[Dict]:
        query_like = f"%{query.strip()}%"

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT course_code, course_name, credits, department, description
                FROM courses
                WHERE course_code LIKE ?
                   OR course_name LIKE ?
                   OR description LIKE ?
                ORDER BY course_code
                LIMIT ?
                """,
                (query_like, query_like, query_like, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_student_profile(self, student_id: str = "1001") -> Dict:
        with self._connect() as connection:
            student = connection.execute(
                """
                SELECT student_id, name, program, semester
                FROM students
                WHERE student_id = ?
                """,
                (student_id,),
            ).fetchone()

            if student is None:
                return {
                    "found": False,
                    "student_id": student_id,
                    "message": f"No student profile found for {student_id}.",
                }

            completed = connection.execute(
                """
                SELECT cc.course_code, c.course_name, c.credits, cc.grade
                FROM completed_courses cc
                JOIN courses c ON c.course_code = cc.course_code
                WHERE cc.student_id = ?
                ORDER BY cc.course_code
                """,
                (student_id,),
            ).fetchall()

        return {
            "found": True,
            "student_id": student["student_id"],
            "name": student["name"],
            "program": student["program"],
            "semester": student["semester"],
            "completed_courses": [dict(row) for row in completed],
        }

    def check_prerequisites(self, student_id: str, course_code: str) -> Dict:
        course_code = normalize_course_code(course_code)

        with self._connect() as connection:
            required_rows = connection.execute(
                """
                SELECT prerequisite_code
                FROM prerequisites
                WHERE course_code = ?
                ORDER BY prerequisite_code
                """,
                (course_code,),
            ).fetchall()
            completed_rows = connection.execute(
                """
                SELECT course_code
                FROM completed_courses
                WHERE student_id = ?
                """,
                (student_id,),
            ).fetchall()

        required = {row["prerequisite_code"] for row in required_rows}
        completed = {row["course_code"] for row in completed_rows}
        missing = sorted(required - completed)

        return {
            "student_id": student_id,
            "course_code": course_code,
            "required_prerequisites": sorted(required),
            "completed_prerequisites": sorted(required & completed),
            "missing_prerequisites": missing,
            "eligible": len(missing) == 0,
        }

    def get_degree_progress(self, student_id: str = "1001") -> Dict:
        profile = self.get_student_profile(student_id)
        if not profile.get("found"):
            return profile

        program = profile["program"]

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    dr.category,
                    dr.required_credits,
                    COALESCE(SUM(c.credits), 0) AS completed_credits
                FROM degree_requirements dr
                LEFT JOIN course_categories cat
                    ON cat.program = dr.program
                   AND cat.category = dr.category
                LEFT JOIN completed_courses cc
                    ON cc.course_code = cat.course_code
                   AND cc.student_id = ?
                LEFT JOIN courses c
                    ON c.course_code = cc.course_code
                WHERE dr.program = ?
                GROUP BY dr.category, dr.required_credits
                ORDER BY dr.category
                """,
                (student_id, program),
            ).fetchall()

        categories = []
        for row in rows:
            completed_credits = row["completed_credits"]
            required_credits = row["required_credits"]
            categories.append(
                {
                    "category": row["category"],
                    "required_credits": required_credits,
                    "completed_credits": completed_credits,
                    "remaining_credits": max(required_credits - completed_credits, 0),
                }
            )

        return {
            "found": True,
            "student_id": student_id,
            "program": program,
            "categories": categories,
            "total_required_credits": sum(item["required_credits"] for item in categories),
            "total_completed_credits": sum(item["completed_credits"] for item in categories),
            "total_remaining_credits": sum(item["remaining_credits"] for item in categories),
        }

    def infer_course_code(self, query: str) -> Optional[str]:
        tokens = query.upper().replace("-", "").split()
        for token in tokens:
            if len(token) >= 4 and token[:2].isalpha() and token[2:].isdigit():
                return token

        compact_query = query.upper().replace("-", "").replace(" ", "")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT course_code, course_name
                FROM courses
                ORDER BY LENGTH(course_name) DESC
                """
            ).fetchall()

        for row in rows:
            if row["course_code"] in compact_query:
                return row["course_code"]
            if row["course_name"].upper() in query.upper():
                return row["course_code"]

        return None

    def infer_student_id(self, query: str, default: str = "1001") -> str:
        for token in query.replace(",", " ").split():
            if token.isdigit() and len(token) >= 4:
                return token
        return default
