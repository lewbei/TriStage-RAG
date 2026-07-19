"""Tests for the RetrievalPipeline."""
import pytest
import yaml
from unittest.mock import patch, MagicMock
from tristage_rag.retrieval_pipeline import RetrievalPipeline, PipelineConfig


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.stage1_model == "google/embeddinggemma-300m"
        assert config.stage2_model == "lightonai/GTE-ModernColBERT-v1"
        assert config.stage3_model == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert config.device == "auto"

    def test_from_yaml(self, tmp_path):
        config_data = {
            "pipeline": {
                "device": "cpu",
                "stage1": {"model": "custom/model1", "top_k": 100},
                "stage2": {"model": "custom/model2"},
                "stage3": {"model": "custom/model3"},
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        pipeline = RetrievalPipeline(config_path=str(config_file))
        assert pipeline.config.device == "cpu"
        assert pipeline.config.stage1_model == "custom/model1"
        assert pipeline.config.stage1_top_k == 100


@pytest.mark.slow
class TestRetrievalPipeline:
    def _make_config(self):
        return PipelineConfig(
            stage1_model="sentence-transformers/all-MiniLM-L6-v2",
            stage2_model="lightonai/GTE-ModernColBERT-v1",
            stage3_model="cross-encoder/ms-marco-MiniLM-L6-v2",
            device="cpu",
            stage1_use_fp16=False,
            stage2_use_fp16=False,
            stage3_use_fp16=False,
        )

    def test_add_documents(self, sample_docs):
        config = self._make_config()
        pipeline = RetrievalPipeline(config=config)
        pipeline.add_documents(sample_docs)
        assert pipeline.stage1 is not None
        assert len(pipeline.stage1.documents) == len(sample_docs)

    def test_search_returns_results(self, sample_docs, sample_query):
        config = self._make_config()
        pipeline = RetrievalPipeline(config=config)
        pipeline.add_documents(sample_docs)
        result = pipeline.search(sample_query, top_k=3)
        assert "query" in result
        assert "results" in result
        assert "timing" in result

    def test_get_pipeline_info(self, sample_docs):
        config = self._make_config()
        pipeline = RetrievalPipeline(config=config)
        pipeline.add_documents(sample_docs)
        info = pipeline.get_pipeline_info()
        assert "config" in info
        assert "stages_initialized" in info
        assert info["stages_initialized"]["stage1"] is True

    def test_empty_search(self):
        config = self._make_config()
        pipeline = RetrievalPipeline(config=config)
        pipeline.add_documents(["test doc"])
        result = pipeline.search("test query")
        assert "results" in result
