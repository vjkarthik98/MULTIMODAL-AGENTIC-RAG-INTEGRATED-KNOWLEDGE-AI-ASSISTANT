from pathlib import Path

import pytest

from app.utils.paths import (
    get_current_user,
    reset_current_user,
    set_current_user,
    user_bm25_path,
    user_dir,
    user_documents_dir,
    user_images_dir,
    user_knowledge_base_dir,
    user_staging_dir,
    user_temp_dir,
    user_temp_frames_dir,
    resolved_staging_dir,
    resolved_temp_dir,
    resolved_images_dir,
    resolved_temp_frames_dir,
)


# ---------------------------------------------------------------------------
# ContextVar helpers
# ---------------------------------------------------------------------------

class TestContextVar:

    def test_set_and_get(self):
        token = set_current_user("user_abc")
        try:
            assert get_current_user() == "user_abc"
        finally:
            reset_current_user(token)

    def test_reset_clears_user(self):
        token = set_current_user("user_abc")
        reset_current_user(token)
        # After reset, value returns to default (None)
        assert get_current_user() is None

    def test_nested_contexts(self):
        t1 = set_current_user("user_1")
        t2 = set_current_user("user_2")
        assert get_current_user() == "user_2"
        reset_current_user(t2)
        assert get_current_user() == "user_1"
        reset_current_user(t1)


# ---------------------------------------------------------------------------
# user_dir
# ---------------------------------------------------------------------------

class TestUserDir:

    def test_returns_path(self, tmp_path):
        token = set_current_user("testuser")
        try:
            p = user_dir("testuser")
            assert isinstance(p, Path)
        finally:
            reset_current_user(token)

    def test_uses_explicit_user_id(self):
        token = set_current_user("testuser")
        try:
            p = user_dir("explicit_user")
            assert "explicit_user" in str(p)
        finally:
            reset_current_user(token)

    def test_uses_contextvar_when_no_arg(self):
        token = set_current_user("ctx_user")
        try:
            p = user_dir()
            assert "ctx_user" in str(p)
        finally:
            reset_current_user(token)

    def test_no_user_raises(self):
        token = set_current_user(None)
        try:
            with pytest.raises(ValueError):
                user_dir()
        finally:
            reset_current_user(token)

    def test_creates_directory(self, tmp_path):
        token = set_current_user("newuser_test")
        try:
            p = user_dir("newuser_test")
            assert p.exists()
        finally:
            reset_current_user(token)


# ---------------------------------------------------------------------------
# Per-user subdirectory helpers
# ---------------------------------------------------------------------------

class TestUserSubdirs:

    def test_knowledge_base_dir_returns_path(self):
        p = user_knowledge_base_dir("u1")
        assert isinstance(p, Path)
        assert p.exists()

    def test_staging_dir_returns_path(self):
        p = user_staging_dir("u1")
        assert isinstance(p, Path)
        assert p.exists()

    def test_documents_dir_returns_path(self):
        p = user_documents_dir("u1")
        assert isinstance(p, Path)
        assert p.exists()

    def test_images_dir_returns_path(self):
        p = user_images_dir("u1")
        assert isinstance(p, Path)
        assert p.exists()

    def test_temp_dir_returns_path(self):
        p = user_temp_dir("u1")
        assert isinstance(p, Path)
        assert p.exists()

    def test_temp_frames_dir_returns_path(self):
        p = user_temp_frames_dir("u1")
        assert isinstance(p, Path)
        assert p.exists()

    def test_subdirs_are_under_user_dir(self):
        base = str(user_dir("u1"))
        assert base in str(user_knowledge_base_dir("u1"))
        assert base in str(user_staging_dir("u1"))
        assert base in str(user_temp_dir("u1"))


# ---------------------------------------------------------------------------
# user_bm25_path
# ---------------------------------------------------------------------------

class TestUserBm25Path:

    def test_returns_path_ending_with_pkl(self):
        p = user_bm25_path("u1")
        assert isinstance(p, Path)
        assert p.name == "bm25.pkl"

    def test_parent_directory_exists(self):
        p = user_bm25_path("u1")
        assert p.parent.exists()

    def test_path_contains_user_id(self):
        p = user_bm25_path("myuser99")
        assert "myuser99" in str(p)


# ---------------------------------------------------------------------------
# Resolved path helpers (contextvar-required)
# ---------------------------------------------------------------------------

class TestResolvedPaths:

    def test_resolved_staging_dir_with_user(self):
        token = set_current_user("ru1")
        try:
            p = resolved_staging_dir()
            assert isinstance(p, Path)
            assert "ru1" in str(p)
        finally:
            reset_current_user(token)

    def test_resolved_staging_dir_without_user_raises(self):
        token = set_current_user(None)
        try:
            with pytest.raises(ValueError):
                resolved_staging_dir()
        finally:
            reset_current_user(token)

    def test_resolved_temp_dir_with_user(self):
        token = set_current_user("ru2")
        try:
            p = resolved_temp_dir()
            assert isinstance(p, Path)
        finally:
            reset_current_user(token)

    def test_resolved_images_dir_with_user(self):
        token = set_current_user("ru3")
        try:
            p = resolved_images_dir()
            assert isinstance(p, Path)
        finally:
            reset_current_user(token)

    def test_resolved_temp_frames_dir_with_user(self):
        token = set_current_user("ru4")
        try:
            p = resolved_temp_frames_dir()
            assert isinstance(p, Path)
        finally:
            reset_current_user(token)
