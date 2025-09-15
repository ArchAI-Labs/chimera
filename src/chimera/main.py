import os
import warnings

from code_explainer.crew import CodeExplainer

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Run the crew.
    """

    inputs = {
        "repository_url": repository_url,
        "repo": repo_to_load,
        "code_path": os.getenv("LOCAL_DIR"),
        "diagram_type": diagram_type,
        "output_format": output_format,
        "sonarqube_json": sonarqube_json,
    }
    try:
        print(f"\n🏗️  Initializing CodeExplainer crew...")
        code_explainer = CodeExplainer()
        
        # Execute analysis
        print(f"\n🔄 Starting analysis...")
        print("=" * 50)
        
        # Optional: Print summary of results
        if isinstance(result, str) and len(result) > 200:
            print(f"\n📋 Analysis Summary (first 200 characters):")
            print(f"   {result[:200]}...")
        
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")