"""
Main entry point for LinkedIn Content Creation Pipeline
Migrated from CrewAI to LlamaIndex
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
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
    output_dir = os.path.join(base_path, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "posts"), exist_ok=True)
    
    return output_dir


def get_user_inputs() -> dict:
    """
    Get user inputs for the content creation pipeline
    
    Returns:
        Dictionary of user inputs
    """
    print("\n" + "="*60)
    print("LinkedIn Content Creation Pipeline (LlamaIndex)")
    print("="*60 + "\n")
    
    # Get expert type
    print("Select expert type:")
    print("1. Generalista (General/Technical content)")
    print("2. Prodotto (Product-focused content with RAG)")
    expert_choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
    
    expert_type = "generalista" if expert_choice == "1" else "prodotto"
    
    # Get topic
    topic = input("\nEnter content topic [default: AI and Technology]: ").strip()
    topic = topic if topic else "AI and Technology"
    
    # Get number of posts
    num_posts_str = input("\nNumber of posts to create [default: 5]: ").strip()
    try:
        num_posts = int(num_posts_str) if num_posts_str else 5
    except ValueError:
        num_posts = 5
    
    # Get posting frequency
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
        print("Set PROVIDER to: ollama, groq, or openai")
        return False
    
    if not model:
        print("\n❌ Error: MODEL not set in .env file")
        print("Examples: llama3, mistral, mixtral (for ollama)")
        return False
    
    # Check provider-specific requirements
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY required for OpenAI provider")
        return False
    
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("\n❌ Error: GROQ_API_KEY required for Groq provider")
        return False
    
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌ Error: ANTHROPIC_API_KEY required for Anthropic provider")
        return False
    
    # Ollama doesn't require API key, just needs to be running
    if provider == "ollama":
        base_url = os.getenv("BASE_URL", "http://localhost:11434")
        print(f"\n✓ Using Ollama at {base_url}")
        print(f"✓ Model: {model}")
        print("Note: Make sure Ollama is running with this model available")
    
    return True


def save_results(results: dict, output_dir: str):
    """
    Save pipeline results to files
    
    Args:
        results: Results dictionary from workflow
        output_dir: Output directory path
    """
    try:
        # Save editorial plan
        if "editorial_plan" in results:
            with open(os.path.join(output_dir, "editorial_plan.md"), "w") as f:
                f.write(results["editorial_plan"])
        
        # Save generated content
        if "generated_content" in results:
            with open(os.path.join(output_dir, "generated_content.md"), "w") as f:
                f.write(results["generated_content"])
        
        # Save LinkedIn posts
        if "linkedin_posts" in results:
            with open(os.path.join(output_dir, "posts", "linkedin_posts.md"), "w") as f:
                f.write(results["linkedin_posts"])
        
        # Save visual assets info
        if "visuals" in results:
            with open(os.path.join(output_dir, "images", "visuals_info.md"), "w") as f:
                f.write(results["visuals"])
        
        # Save final plan
        if "final_plan" in results:
            with open(os.path.join(output_dir, "final_content_plan.md"), "w") as f:
                f.write(results["final_plan"])
        
        # Save complete results
        with open(os.path.join(output_dir, "complete_results.txt"), "w") as f:
            f.write(str(results))
        
        print(f"\n✅ Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n⚠️  Error saving results: {str(e)}")


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
        
        # Create crew instance
        crew = LinkedInCrew(inputs=inputs)
        
        print("\n🚀 Starting content generation workflow...\n")
        
        # Run the workflow
        results = await crew.kickoff()
        
        print("\n" + "="*60)
        print("Pipeline Execution Complete!")
        print("="*60 + "\n")
        
        # Save results
        save_results(results, output_dir)
        
        # Print summary
        if results.get("status") == "success":
            print("\n✅ Content creation successful!")
            print(f"\n📁 Output directory: {output_dir}")
            print("\nGenerated files:")
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
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Get user inputs
        inputs = get_user_inputs()
        
        # Create output directory
        output_dir = create_output_directory()
        
        print(f"\n📝 Configuration:")
        print(f"   Expert Type: {inputs['expert_type']}")
        print(f"   Topic: {inputs['topic']}")
        print(f"   Number of Posts: {inputs['num_posts']}")
        print(f"   Frequency: {inputs['frequency']}")
        print(f"   Output Directory: {output_dir}")
        
        # Confirm before proceeding
        confirm = input("\n▶️  Proceed with content generation? (y/n) [default: y]: ").strip().lower()
        if confirm and confirm != 'y':
            print("\n❌ Operation cancelled.")
            sys.exit(0)
        
        # Run pipeline
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