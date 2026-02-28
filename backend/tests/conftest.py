import sys
import os
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


@pytest.fixture
def real_vector_store(tmp_path):
    from vector_store import VectorStore
    return VectorStore(str(tmp_path / "chroma"), "all-MiniLM-L6-v2", max_results=5)


@pytest.fixture
def mock_vector_store(mocker):
    return mocker.MagicMock()


@pytest.fixture
def sample_chunk_no_lesson():
    from models import CourseChunk
    return CourseChunk(
        content="General content.",
        course_title="Test Course",
        lesson_number=None,
        chunk_index=0,
    )


@pytest.fixture
def sample_course_no_optionals():
    from models import Course
    return Course(title="Test Course", instructor=None, course_link=None)
