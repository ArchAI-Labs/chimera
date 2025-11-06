########### UTF-8 SETUP - MUST BE FIRST #####################
import sys
import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

##############################################################

from datetime import datetime
from pathlib import Path
import asyncio
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent))

from crew_llamaindex import LinkedInCrew


def create_output_directory(base_path: str = "output") -> str:
    """
    Create timestamped output directory
    
    Args:
        base_path: Base directory for outputs
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "posts").mkdir(exist_ok=True)
    
    return str(output_dir)


def get_user_inputs() -> dict:
    """
    Get user inputs for the content creation pipeline
    
    Returns:
        Dictionary of user inputs
    """
    print("\n" + "="*60)
    print("LinkedIn Content Creation Pipeline (LlamaIndex)")
    print("="*60 + "\n")
    
    print("Select expert type:")
    print("1. Generalista (General/Technical content)")
    print("2. Prodotto (Product-focused content with RAG)")
    expert_choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    expert_type = "generalista" if expert_choice == "1" else "prodotto"
    
    topic = input("\nEnter content topic [default: AI and Technology]: ").strip()
    topic = topic if topic else "AI and Technology"
    
    num_posts_str = input("\nNumber of posts to create [default: 5]: ").strip()
    try:
        num_posts = int(num_posts_str) if num_posts_str else 5
    except ValueError:
        num_posts = 5
    
    print("\nPosting frequency:")
    print("1. Daily")
    print("2. Weekly")
    print("3. Bi-weekly")
    print("4. Monthly")
    frequency_choice = input("\nEnter choice (1-4) [default: 2]: ").strip() or "2"
    
    frequency_map = {
        "1": "daily",
        "2": "weekly",
        "3": "bi-weekly",
        "4": "monthly"
    }
    frequency = frequency_map.get(frequency_choice, "weekly")
    
    return {
        "expert_type": expert_type,
        "topic": topic,
        "num_posts": num_posts,
        "frequency": frequency,
    }


def validate_environment() -> bool:
    """
    Validate required environment variables
    
    Returns:
        True if environment is valid, False otherwise
    """
    provider = os.getenv("PROVIDER", "").lower()
    model = os.getenv("MODEL")
    
    if not provider:
        print("\n❌ Error: PROVIDER not set in .env file")
        print("Set PROVIDER to: ollama, groq, openai, or anthropic")
        return False
    
    if not model:
        print("\n❌ Error: MODEL not set in .env file")
        print("Examples: llama3, mistral, mixtral (for ollama)")
        return False
    
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY required for OpenAI provider")
        return False
    
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("\n❌ Error: GROQ_API_KEY required for Groq provider")
        return False
    
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ Error: ANTHROPIC_API_KEY required for Anthropic provider")
        return False
    
    if provider == "ollama":
        base_url = os.getenv("BASE_URL", "http://localhost:11434")
        print(f"\n✓ Using Ollama at {base_url}")
        print(f"✓ Model: {model}")
        print("Note: Make sure Ollama is running with this model available")
    
    return True


def save_text_file_utf8(filepath: Path, content: str):
    """
    Save text file with proper UTF-8 encoding (handles emojis and special chars).
    
    Args:
        filepath: Path to file
        content: Text content to write
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        utf8_bytes = content.encode('utf-8', errors='replace')
        f.write(utf8_bytes)


def save_results(results: dict, output_dir: str):
    """
    Save pipeline results to files with proper UTF-8 encoding.
    
    Args:
        results: Results dictionary from workflow
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    
    try:
        if "editorial_plan" in results:
            save_text_file_utf8(
                output_path / "editorial_plan.md",
                results["editorial_plan"]
            )
        
        if "generated_content" in results:
            save_text_file_utf8(
                output_path / "generated_content.md",
                results["generated_content"]
            )
        
        if "linkedin_posts" in results:
            save_text_file_utf8(
                output_path / "posts" / "linkedin_posts.md",
                results["linkedin_posts"]
            )
        
        if "visuals" in results:
            save_text_file_utf8(
                output_path / "images" / "visuals_info.md",
                results["visuals"]
            )
        
        if "final_plan" in results:
            final_plan_content = results.get("final_plan", "")
            if isinstance(final_plan_content, str) and len(final_plan_content) > 100:
                save_text_file_utf8(
                    output_path / "final_content_plan.md",
                    final_plan_content
                )
        
        results_text = format_results_summary(results)
        save_text_file_utf8(
            output_path / "complete_results.txt",
            results_text
        )
        
        print(f"\n✅ Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n⚠️  Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()


def format_results_summary(results: dict) -> str:
    """
    Format results dictionary into a readable text summary.
    
    Args:
        results: Results dictionary
        
    Returns:
        Formatted text summary
    """
    lines = []
    lines.append("="*60)
    lines.append("LINKEDIN CONTENT CREATION RESULTS")
    lines.append("="*60)
    lines.append("")
    
    lines.append(f"Status: {results.get('status', 'unknown')}")
    lines.append(f"Output Directory: {results.get('output_directory', 'N/A')}")
    lines.append("")
    
    if "editorial_plan" in results:
        lines.append("-" * 60)
        lines.append("EDITORIAL PLAN")
        lines.append("-" * 60)
        lines.append(results["editorial_plan"][:500] + "..." if len(results["editorial_plan"]) > 500 else results["editorial_plan"])
        lines.append("")
    
    if "generated_content" in results:
        lines.append("-" * 60)
        lines.append("GENERATED CONTENT")
        lines.append("-" * 60)
        lines.append(results["generated_content"][:500] + "..." if len(results["generated_content"]) > 500 else results["generated_content"])
        lines.append("")
    
    if "linkedin_posts" in results:
        lines.append("-" * 60)
        lines.append("LINKEDIN POSTS")
        lines.append("-" * 60)
        lines.append(results["linkedin_posts"][:500] + "..." if len(results["linkedin_posts"]) > 500 else results["linkedin_posts"])
        lines.append("")
    
    if "visuals" in results:
        lines.append("-" * 60)
        lines.append("VISUAL CONCEPTS")
        lines.append("-" * 60)
        lines.append(results["visuals"][:500] + "..." if len(results["visuals"]) > 500 else results["visuals"])
        lines.append("")
    
    lines.append("="*60)
    lines.append("END OF RESULTS")
    lines.append("="*60)
    
    return "\n".join(lines)


async def run_pipeline_async(inputs: dict, output_dir: str):
    """
    Run the content creation pipeline asynchronously
    
    Args:
        inputs: User inputs dictionary
        output_dir: Output directory path
    """
    try:
        print("\n" + "="*60)
        print("Initializing LinkedIn Content Pipeline...")
        print("="*60 + "\n")
        

        inputs["output_dir"] = output_dir
        

        crew = LinkedInCrew(inputs=inputs)
        
        print("\n🚀 Starting content generation workflow...\n")

        results = await crew.kickoff()
        
        print("\n" + "="*60)
        print("Pipeline Execution Complete!")
        print("="*60 + "\n")
        
        save_results(results, output_dir)
        
        if results.get("status") == "success":
            print("\n✅ Content creation successful!")
            print(f"\n📁 Output directory: {output_dir}")
            print("\nGenerated files:")

            output_path = Path(output_dir)
            md_files = sorted(output_path.rglob("*.md"))
            
            if md_files:
                for md_file in md_files:
                    relative_path = md_file.relative_to(output_path)
                    print(f"  - {relative_path}")
            else:
                print("  - editorial_plan.md")
                print("  - generated_content.md")
                print("  - posts/linkedin_posts.md")
                print("  - images/visuals_info.md")
                print("  - final_content_plan.md")
        else:
            print("\n⚠️  Pipeline completed with warnings. Check output files.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def main():
    """
    Main entry point
    """
    try:
        if not validate_environment():
            sys.exit(1)

        inputs = get_user_inputs()

        output_dir = create_output_directory()
        
        print(f"\n📝 Configuration:")
        print(f"   Expert Type: {inputs['expert_type']}")
        print(f"   Topic: {inputs['topic']}")
        print(f"   Number of Posts: {inputs['num_posts']}")
        print(f"   Frequency: {inputs['frequency']}")
        print(f"   Output Directory: {output_dir}")
        
        confirm = input("\n▶️  Proceed with content generation? (y/n) [default: y]: ").strip().lower()
        if confirm and confirm != 'y':
            print("\n❌ Operation cancelled.")
            sys.exit(0)
        
        results = asyncio.run(run_pipeline_async(inputs, output_dir))
        
        if results.get("status") == "error":
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n❌ Operation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()