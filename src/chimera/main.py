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
import json
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

load_dotenv()

sys.path.append(str(Path(__file__).parent))

from crew_llamaindex import LinkedInCrew


def get_llm():
    provider = os.getenv("PROVIDER", "").lower()
    model = os.getenv("MODEL")
    
    if provider == "ollama":
        return Ollama(model=model, request_timeout=120.0)
    elif provider == "openai":
        return OpenAI(model=model)
    elif provider == "anthropic":
        return Anthropic(model=model)
    else:
        return Ollama(model="llama3", request_timeout=120.0)


def extract_info_with_llm(user_input: str, collected: dict, llm) -> dict:
    """Use LLM to extract information from user input in a single call."""
    
    system_prompt = f"""You are extracting information for a LinkedIn content pipeline.

Current state:
- expert_type: {collected['expert_type']} (need: "generalista" or "prodotto")
- topic: {collected['topic']} (need: string)
- num_posts: {collected['num_posts']} (need: integer)
- frequency: {collected['frequency']} (need: "daily", "weekly", "bi-weekly", or "monthly")

Extract any NEW information from the user's message. Map keywords:
- "general", "technical", "generalista" → "generalista"
- "product", "prodotto" → "prodotto"
- For topics: extract the main subject
- For numbers: extract integers
- For frequency: "daily", "weekly", "bi-weekly", "monthly"

Return ONLY valid JSON:
{{
  "expert_type": null or "generalista" or "prodotto",
  "topic": null or "string",
  "num_posts": null or integer,
  "frequency": null or "daily"/"weekly"/"bi-weekly"/"monthly"
}}

If nothing can be extracted, return all nulls."""

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_input)
    ]
    
    try:
        response = llm.chat(messages)
        response_text = response.message.content.strip()
        
        # Extract JSON
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "{" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
        else:
            return {}
        
        extracted = json.loads(json_str)
        return extracted
        
    except Exception as e:
        print(f"[Debug: Extraction failed - {e}]")
        return {}


def get_user_inputs() -> dict:
    
    print("\n" + "="*60)
    print("LinkedIn Content Creation Pipeline")
    print("="*60)
    print("\n👋 Hi! I'm your LinkedIn content assistant.")
    print("\nI need to collect:")
    print("  • Expert type (General/Technical or Product-focused)")
    print("  • Content topic")
    print("  • Number of posts")
    print("  • Posting frequency")
    print("\nFeel free to tell me everything at once or step by step!\n")
    
    llm = get_llm()
    
    collected = {
        "expert_type": None,
        "topic": None,
        "num_posts": None,
        "frequency": None
    }
    
    # Conversation history for context
    conversation = []
    
    max_turns = 10
    turn = 0
    
    while turn < max_turns:
        # Check if we have everything
        if all(v is not None for v in collected.values()):
            break
        
        # Ask for input
        if turn == 0:
            prompt = "You: "
        else:
            # Determine what to ask for
            missing = [k for k, v in collected.items() if v is None]
            if missing:
                field_names = {
                    "expert_type": "expert type (generalista or prodotto)",
                    "topic": "topic",
                    "num_posts": "number of posts",
                    "frequency": "posting frequency (daily/weekly/bi-weekly/monthly)"
                }
                print(f"\nI still need: {field_names[missing[0]]}")
                prompt = "You: "
            else:
                prompt = "You: "
        
        user_input = input(prompt).strip()
        
        if not user_input:
            print("Please provide some information.\n")
            continue
        
        conversation.append(user_input)
        
        extracted = extract_info_with_llm(user_input, collected, llm)
        
        updated_fields = []
        for key, value in extracted.items():
            if value is not None and key in collected and collected[key] is None:
                collected[key] = value
                updated_fields.append(f"{key}: {value}")
        
        if updated_fields:
            for field in updated_fields:
                print(f"  ✓ Got {field}")
        else:
            print("  (Couldn't extract specific info from that)")
        
        turn += 1
    
    if collected["expert_type"] is None:
        print("\n⚠️  Expert type not specified, using default: generalista")
        collected["expert_type"] = "generalista"
    
    if collected["topic"] is None:
        print("⚠️  Topic not specified, using default: AI and Technology")
        collected["topic"] = "AI and Technology"
    
    if collected["num_posts"] is None:
        print("⚠️  Number of posts not specified, using default: 5")
        collected["num_posts"] = 5
    
    if collected["frequency"] is None:
        print("⚠️  Frequency not specified, using default: weekly")
        collected["frequency"] = "weekly"
    
    print(f"\n{'='*60}")
    print("Collected Information:")
    print(f"  • Expert Type: {collected['expert_type']}")
    print(f"  • Topic: {collected['topic']}")
    print(f"  • Number of Posts: {collected['num_posts']}")
    print(f"  • Frequency: {collected['frequency']}")
    print(f"{'='*60}\n")
    
    confirm = input("Is this correct? (Y/n): ").strip().lower()
    
    if confirm and confirm not in ["y", "yes", ""]:
        print("\nLet's start over...\n")
        return get_user_inputs()  # Recursive call to restart
    
    return collected


def create_output_directory(base_path: str = "output") -> str:
    """Create timestamped output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "posts").mkdir(exist_ok=True)
    
    return str(output_dir)


def validate_environment() -> bool:
    """Validate environment variables and configuration."""
    provider = os.getenv("PROVIDER", "").lower()
    model = os.getenv("MODEL")
    
    if not provider:
        print("\n❌ Error: PROVIDER not set in .env file")
        print("Set PROVIDER to: ollama, openai, or anthropic")
        return False
    
    if not model:
        print("\n❌ Error: MODEL not set in .env file")
        print("Examples: llama3, mistral, mixtral (for ollama)")
        return False
    
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY required for OpenAI provider")
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
    """Save text content to file with UTF-8 encoding."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        utf8_bytes = content.encode('utf-8', errors='replace')
        f.write(utf8_bytes)


def save_results(results: dict, output_dir: str):
    """Save all results to output directory."""
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
    """Format results into a readable summary."""
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
    """Run the LinkedIn content creation pipeline."""
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
    """Main entry point."""
    try:
        if not validate_environment():
            sys.exit(1)
        
        inputs = get_user_inputs()
        
        output_dir = create_output_directory()
        
        print(f"\n📝 Final Configuration:")
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