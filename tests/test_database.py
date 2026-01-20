from utils.database import get_db_path, get_model_dim
import pytest 

class TestGetDBPath:
    """
    Test the get_db_path function.
    """
    def test_model_with_meta(self):
        """Standard model without metadata."""
        path = get_db_path("all-MiniLM-L6-v2", include_meta=True, db_tag="")
        assert path == "./vector_db_all-MiniLM-L6-v2.db"

    def test_model_without_meta(self):
        """Standard model without metadata."""
        path = get_db_path("all-MiniLM-L6-v2", include_meta=False, db_tag="")
        assert path == "./vector_db_all-MiniLM-L6-v2_no_meta.db"

    def test_all_options(self):
        """All options: namespace + tag + meta."""
        path = get_db_path("jxm/cde-small-v2", include_meta=True, db_tag="x2")
        assert path == "./vector_db_cde-small-v2-x2.db"
        
class TestModelDim:
    """
    Test the get_model_dim function.
    """
    
    def test_get_model_dim_valid(self):
        dim = get_model_dim("all-MiniLM-L6-v2")
        assert dim == 384

    def test_get_model_dim_invalid(self):
        with pytest.raises(KeyError):
            get_model_dim("invalid-model")
