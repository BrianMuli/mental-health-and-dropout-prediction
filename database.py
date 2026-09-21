import sqlite3
from pathlib import Path
from datetime import datetime

# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "students.db"


def get_connection():
    """Create and return a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    """Create students and assessments tables if they don't exist."""

    conn = get_connection()
    cursor = conn.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            department TEXT,
            year_of_study INTEGER,
            gpa REAL,
            semester_gpa REAL,
            cgpa REAL,
            attendance_rate REAL,
            study_hours_per_day REAL,
            assignment_delay_days REAL,
            travel_time_minutes REAL,
            family_income REAL,
            scholarship TEXT,
            part_time_job TEXT,
            internet_access TEXT,
            parental_education TEXT
        )
    """)

    # Assessments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            stress_index REAL,
            dropout_probability REAL,
            risk_level TEXT,
            assessment_date TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    conn.commit()
    conn.close()


def save_student(student):
    """Insert or update a student profile."""

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO students (
            student_id,
            age,
            gender,
            department,
            year_of_study,
            gpa,
            semester_gpa,
            cgpa,
            attendance_rate,
            study_hours_per_day,
            assignment_delay_days,
            travel_time_minutes,
            family_income,
            scholarship,
            part_time_job,
            internet_access,
            parental_education
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student["student_id"],
        student["age"],
        student["gender"],
        student["department"],
        student["year_of_study"],
        student["gpa"],
        student["semester_gpa"],
        student["cgpa"],
        student["attendance_rate"],
        student["study_hours_per_day"],
        student["assignment_delay_days"],
        student["travel_time_minutes"],
        student["family_income"],
        student["scholarship"],
        student["part_time_job"],
        student["internet_access"],
        student["parental_education"]
    ))

    conn.commit()
    conn.close()


def get_student(student_id):
    """Retrieve a student profile."""

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM students WHERE student_id = ?",
        (student_id,)
    )

    student = cursor.fetchone()

    conn.close()

    return dict(student) if student else None


def save_assessment(
    student_id,
    stress_index,
    dropout_probability,
    risk_level
):
    """Save the result of an ML assessment."""

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO assessments (
            student_id,
            stress_index,
            dropout_probability,
            risk_level,
            assessment_date
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        student_id,
        stress_index,
        dropout_probability,
        risk_level,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_assessment_history(student_id):
    """Retrieve previous assessments for a student."""

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM assessments
        WHERE student_id = ?
        ORDER BY assessment_date DESC
    """, (student_id,))

    assessments = cursor.fetchall()

    conn.close()

    return [dict(row) for row in assessments]