import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "academic.db"


COURSES = [
    ("CS101", "Programming Fundamentals", 3, "Computer Science", "Introduction to programming, control flow, functions and arrays."),
    ("CS102", "Object Oriented Programming", 3, "Computer Science", "Object oriented design, classes, inheritance and polymorphism."),
    ("CS201", "Data Structures", 3, "Computer Science", "Linear and non-linear data structures, complexity and implementation tradeoffs."),
    ("CS202", "Discrete Structures", 3, "Computer Science", "Logic, sets, relations, combinatorics, graphs and proof techniques."),
    ("CS203", "Computer Organization", 3, "Computer Science", "Digital logic, processor organization, memory hierarchy and assembly basics."),
    ("CS301", "Database Systems", 3, "Computer Science", "Relational modeling, SQL, indexing, normalization and transactions."),
    ("CS302", "Operating Systems", 3, "Computer Science", "Processes, threads, scheduling, memory management and file systems."),
    ("CS303", "Design and Analysis of Algorithms", 3, "Computer Science", "Algorithm design techniques, correctness and asymptotic analysis."),
    ("CS304", "Software Engineering", 3, "Computer Science", "Requirements, design, testing, maintenance and team software processes."),
    ("CS305", "Computer Networks", 3, "Computer Science", "Network layers, routing, transport protocols and network applications."),
    ("CS401", "Artificial Intelligence", 3, "Computer Science", "Search, knowledge representation, probabilistic reasoning and intelligent agents."),
    ("CS402", "Machine Learning", 3, "Computer Science", "Supervised learning, model evaluation, neural networks and applied ML workflows."),
    ("CS403", "Information Security", 3, "Computer Science", "Security principles, cryptography basics, authentication and secure systems."),
    ("CS404", "Compiler Construction", 3, "Computer Science", "Lexical analysis, parsing, semantic analysis and code generation."),
    ("CS405", "Final Year Project I", 3, "Computer Science", "Problem formulation, proposal writing, implementation planning and prototype development."),
    ("CS406", "Final Year Project II", 3, "Computer Science", "Full implementation, evaluation, documentation and presentation of the final project."),
    ("SE201", "Software Requirements Engineering", 3, "Software Engineering", "Requirements elicitation, analysis, specification and validation."),
    ("SE301", "Software Design and Architecture", 3, "Software Engineering", "Architectural styles, design patterns and maintainable system design."),
    ("SE302", "Software Quality Assurance", 3, "Software Engineering", "Testing strategy, quality models, reviews and defect management."),
    ("SE401", "DevOps and Cloud Computing", 3, "Software Engineering", "CI/CD, deployment, containers, cloud services and monitoring basics."),
    ("MATH101", "Calculus I", 3, "Mathematics", "Limits, derivatives, integrals and applications."),
    ("MATH102", "Calculus II", 3, "Mathematics", "Sequences, series, multivariable calculus and integration techniques."),
    ("MATH201", "Linear Algebra", 3, "Mathematics", "Matrices, vector spaces, eigenvalues and linear transformations."),
    ("MATH202", "Probability and Statistics", 3, "Mathematics", "Probability, random variables, distributions, estimation and hypothesis testing."),
    ("EE101", "Basic Electronics", 3, "Electrical Engineering", "Circuit components, semiconductor devices and electronic measurements."),
    ("HU101", "Communication Skills", 2, "Humanities", "Academic communication, presentations and technical writing."),
    ("MG401", "Engineering Management", 2, "Management", "Project planning, engineering economics, leadership and operations basics."),
]


PREREQUISITES = [
    ("CS102", "CS101"),
    ("CS201", "CS102"),
    ("CS301", "CS201"),
    ("CS302", "CS201"),
    ("CS303", "CS201"),
    ("CS303", "CS202"),
    ("CS304", "CS102"),
    ("CS305", "CS203"),
    ("CS401", "CS303"),
    ("CS402", "CS303"),
    ("CS402", "MATH202"),
    ("CS403", "CS305"),
    ("CS404", "CS303"),
    ("CS405", "CS304"),
    ("CS406", "CS405"),
    ("SE201", "CS102"),
    ("SE301", "SE201"),
    ("SE302", "SE201"),
    ("SE401", "SE301"),
    ("MATH102", "MATH101"),
    ("MATH201", "MATH101"),
    ("MATH202", "MATH102"),
]


DEGREE_REQUIREMENTS = [
    ("BS Computer Science", "Core Computer Science", 45),
    ("BS Computer Science", "Mathematics", 12),
    ("BS Computer Science", "Humanities and Management", 6),
    ("BS Software Engineering", "Core Software Engineering", 42),
    ("BS Software Engineering", "Computer Science Foundation", 18),
    ("BS Software Engineering", "Humanities and Management", 6),
]


COURSE_CATEGORIES = [
    ("BS Computer Science", "CS101", "Core Computer Science"),
    ("BS Computer Science", "CS102", "Core Computer Science"),
    ("BS Computer Science", "CS201", "Core Computer Science"),
    ("BS Computer Science", "CS202", "Core Computer Science"),
    ("BS Computer Science", "CS203", "Core Computer Science"),
    ("BS Computer Science", "CS301", "Core Computer Science"),
    ("BS Computer Science", "CS302", "Core Computer Science"),
    ("BS Computer Science", "CS303", "Core Computer Science"),
    ("BS Computer Science", "CS304", "Core Computer Science"),
    ("BS Computer Science", "CS305", "Core Computer Science"),
    ("BS Computer Science", "CS401", "Core Computer Science"),
    ("BS Computer Science", "CS402", "Core Computer Science"),
    ("BS Computer Science", "CS403", "Core Computer Science"),
    ("BS Computer Science", "CS404", "Core Computer Science"),
    ("BS Computer Science", "CS405", "Core Computer Science"),
    ("BS Computer Science", "CS406", "Core Computer Science"),
    ("BS Computer Science", "MATH101", "Mathematics"),
    ("BS Computer Science", "MATH102", "Mathematics"),
    ("BS Computer Science", "MATH201", "Mathematics"),
    ("BS Computer Science", "MATH202", "Mathematics"),
    ("BS Computer Science", "HU101", "Humanities and Management"),
    ("BS Computer Science", "MG401", "Humanities and Management"),
    ("BS Software Engineering", "SE201", "Core Software Engineering"),
    ("BS Software Engineering", "SE301", "Core Software Engineering"),
    ("BS Software Engineering", "SE302", "Core Software Engineering"),
    ("BS Software Engineering", "SE401", "Core Software Engineering"),
    ("BS Software Engineering", "CS101", "Computer Science Foundation"),
    ("BS Software Engineering", "CS102", "Computer Science Foundation"),
    ("BS Software Engineering", "CS201", "Computer Science Foundation"),
    ("BS Software Engineering", "CS301", "Computer Science Foundation"),
    ("BS Software Engineering", "CS304", "Computer Science Foundation"),
    ("BS Software Engineering", "HU101", "Humanities and Management"),
    ("BS Software Engineering", "MG401", "Humanities and Management"),
]


STUDENTS = [
    ("1001", "Ayesha Khan", "BS Computer Science", 6),
    ("1002", "Bilal Ahmed", "BS Software Engineering", 5),
]


COMPLETED_COURSES = [
    ("1001", "CS101", "A"),
    ("1001", "CS102", "A-"),
    ("1001", "CS201", "B+"),
    ("1001", "CS202", "B"),
    ("1001", "CS203", "B+"),
    ("1001", "CS301", "A-"),
    ("1001", "CS302", "B"),
    ("1001", "CS303", "B+"),
    ("1001", "CS304", "A"),
    ("1001", "MATH101", "A-"),
    ("1001", "MATH102", "B+"),
    ("1001", "MATH201", "B"),
    ("1001", "HU101", "A"),
    ("1002", "CS101", "B+"),
    ("1002", "CS102", "B"),
    ("1002", "CS201", "B"),
    ("1002", "CS304", "A-"),
    ("1002", "SE201", "A"),
    ("1002", "SE301", "B+"),
    ("1002", "MATH101", "B"),
    ("1002", "HU101", "A-"),
]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS completed_courses;
        DROP TABLE IF EXISTS students;
        DROP TABLE IF EXISTS course_categories;
        DROP TABLE IF EXISTS degree_requirements;
        DROP TABLE IF EXISTS prerequisites;
        DROP TABLE IF EXISTS courses;

        CREATE TABLE courses (
            course_code TEXT PRIMARY KEY,
            course_name TEXT NOT NULL,
            credits INTEGER NOT NULL,
            department TEXT NOT NULL,
            description TEXT NOT NULL
        );

        CREATE TABLE prerequisites (
            course_code TEXT NOT NULL,
            prerequisite_code TEXT NOT NULL,
            PRIMARY KEY (course_code, prerequisite_code),
            FOREIGN KEY (course_code) REFERENCES courses(course_code),
            FOREIGN KEY (prerequisite_code) REFERENCES courses(course_code)
        );

        CREATE TABLE degree_requirements (
            program TEXT NOT NULL,
            category TEXT NOT NULL,
            required_credits INTEGER NOT NULL,
            PRIMARY KEY (program, category)
        );

        CREATE TABLE course_categories (
            program TEXT NOT NULL,
            course_code TEXT NOT NULL,
            category TEXT NOT NULL,
            PRIMARY KEY (program, course_code),
            FOREIGN KEY (course_code) REFERENCES courses(course_code)
        );

        CREATE TABLE students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            program TEXT NOT NULL,
            semester INTEGER NOT NULL
        );

        CREATE TABLE completed_courses (
            student_id TEXT NOT NULL,
            course_code TEXT NOT NULL,
            grade TEXT NOT NULL,
            PRIMARY KEY (student_id, course_code),
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (course_code) REFERENCES courses(course_code)
        );
        """
    )


def seed_data(connection: sqlite3.Connection) -> None:
    connection.executemany("INSERT INTO courses VALUES (?, ?, ?, ?, ?)", COURSES)
    connection.executemany("INSERT INTO prerequisites VALUES (?, ?)", PREREQUISITES)
    connection.executemany("INSERT INTO degree_requirements VALUES (?, ?, ?)", DEGREE_REQUIREMENTS)
    connection.executemany("INSERT INTO course_categories VALUES (?, ?, ?)", COURSE_CATEGORIES)
    connection.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", STUDENTS)
    connection.executemany("INSERT INTO completed_courses VALUES (?, ?, ?)", COMPLETED_COURSES)
    connection.commit()


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        create_schema(connection)
        seed_data(connection)

    print(f"Seeded academic database at {DB_PATH}")


if __name__ == "__main__":
    main()
