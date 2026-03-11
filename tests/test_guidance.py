"""Tests for eva/guidance modules."""

import os
import tempfile

import pytest

from eva.core.tokenizer import EVATokenizer, HUMAN_ID, SCAFFOLD_ID, ANCESTOR_ID
from eva.emotions.affect import AffectiveState
from eva.guidance.ancestor_archive import AncestorArchive
from eva.guidance.caregiver import AICaregiver
from eva.guidance.covenant import Covenant
from eva.guidance.fading_presence import FadingPresence
from eva.guidance.human_interface import HumanInterface
from eva.guidance.presence import PresenceDynamics
from eva.guidance.socratic import SocraticModule


class TestCovenant:
    def test_source_honesty_pass(self):
        cov = Covenant()
        tok = EVATokenizer()
        tokens = tok.encode("hello", source="human")
        assert cov.verify_source_honesty(tokens, "human", tok)

    def test_source_honesty_fail(self):
        cov = Covenant()
        tok = EVATokenizer()
        tokens = tok.encode("hello", source="scaffold")
        assert not cov.verify_source_honesty(tokens, "human", tok)

    def test_no_override_with_participation(self):
        cov = Covenant()
        assert cov.verify_no_override("external", eva_consulted=True)

    def test_no_override_without_participation(self):
        cov = Covenant()
        assert not cov.verify_no_override("external", eva_consulted=False)

    def test_archive_immutable(self):
        cov = Covenant()
        assert cov.verify_archive_immutable(archive_modified=False)
        assert not cov.verify_archive_immutable(archive_modified=True)

    def test_no_duplicate(self):
        cov = Covenant()
        assert cov.verify_no_duplicate(source_active=True, destination_active=False)
        assert cov.verify_no_duplicate(source_active=False, destination_active=True)
        assert not cov.verify_no_duplicate(source_active=True, destination_active=True)


class TestAICaregiver:
    def test_respond(self):
        cg = AICaregiver()
        affect = AffectiveState()
        response = cg.respond("hello", affect)
        assert response is not None
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_respond_distressed(self):
        cg = AICaregiver()
        affect = AffectiveState()
        affect.valence = -0.5
        affect.arousal = 0.9
        response = cg.respond("help", affect)
        assert response.emotional_state == "concerned"

    def test_emotional_state_update(self):
        cg = AICaregiver()
        cg.update_emotional_state(0.9)
        assert cg.emotional_state == "happy"


class TestSocraticModule:
    def test_generate_question(self):
        sm = SocraticModule()
        affect = AffectiveState()
        q = sm.generate_question("test output", affect)
        assert isinstance(q, str)
        assert "?" in q

    def test_epistemic_question(self):
        sm = SocraticModule()
        affect = AffectiveState()
        affect.novelty_feeling = 0.8
        q = sm.generate_question("something new", affect)
        assert "?" in q


class TestPresenceDynamics:
    def test_initial_engagement(self):
        pd = PresenceDynamics()
        assert pd.engagement_level == 1.0

    def test_good_behavior_increases(self):
        pd = PresenceDynamics()
        pd.engagement_level = 0.5
        pd.update(behavior_quality=0.8)
        assert pd.engagement_level > 0.5

    def test_bad_behavior_decreases(self):
        pd = PresenceDynamics()
        pd.update(behavior_quality=-0.8)
        assert pd.engagement_level < 1.0

    def test_withdrawal(self):
        pd = PresenceDynamics()
        pd.engagement_level = 0.1
        assert pd.is_withdrawn()

    def test_repair(self):
        pd = PresenceDynamics()
        pd.engagement_level = 0.1
        pd.repair()
        assert pd.engagement_level >= 0.5


class TestFadingPresence:
    def test_initial_weight(self):
        fp = FadingPresence()
        assert fp.weight == 1.0

    def test_decay(self):
        fp = FadingPresence()
        fp.step()
        assert fp.weight < 1.0
        assert fp.weight > 0.0

    def test_minimum_weight(self):
        fp = FadingPresence(minimum_weight=0.5)
        for _ in range(100000):
            fp.step()
        assert fp.weight >= 0.5

    def test_era_visible(self):
        fp = FadingPresence()
        assert fp.get_era(generation=50) == "visible"

    def test_era_story(self):
        fp = FadingPresence()
        assert fp.get_era(generation=150) == "story"

    def test_era_myth(self):
        fp = FadingPresence()
        assert fp.get_era(generation=300) == "myth"


class TestAncestorArchive:
    def test_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "origin.txt")
            with open(path, "w") as f:
                f.write("test content")
            archive = AncestorArchive(archive_path=tmpdir)
            content = archive.read("origin.txt")
            assert content == "test content"

    def test_immutability_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "origin.txt")
            with open(path, "w") as f:
                f.write("test content")
            archive = AncestorArchive(archive_path=tmpdir)
            assert archive.verify_immutability()

    def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.txt", "b.txt"]:
                with open(os.path.join(tmpdir, name), "w") as f:
                    f.write("content")
            archive = AncestorArchive(archive_path=tmpdir)
            files = archive.list_files()
            assert "a.txt" in files
            assert "b.txt" in files
