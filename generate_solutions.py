"""
AI-Powered Solution Generator for Mathematics Questions
========================================================

Leverages Large Language Models to generate detailed, pedagogically sound
step-by-step solutions for mathematics questions across multiple topics.

Features:
- Integration with Groq API for fast LLM inference
- Structured prompting for consistent solution quality
- Student-friendly explanation generation
- Multi-format output (JSON and human-readable text)

Model: Llama 3.3 (70B parameters) via Groq API
Target Audience: High school mathematics students
Project: CSI Club VIT Vellore Selection Task (Bonus Component)
"""

import json
from groq import Groq
from pathlib import Path
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_groq():
    """
    Initialize connection to Groq API service.
    
    Groq provides access to high-performance LLM inference with models
    including Llama 3.3. The API key should be stored in a .env file
    for security purposes.
    
    Environment Variable Required:
        GROQ_API_KEY: Authentication token from console.groq.com
        
    Returns:
        Groq: Authenticated API client
        None: If API key is not found or invalid
        
    Setup Instructions:
        1. Visit https://console.groq.com/
        2. Create a free account
        3. Generate an API key
        4. Create .env file: GROQ_API_KEY=your_key_here
    """
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        print("\nSetup Instructions:")
        print("1. Visit https://console.groq.com/")
        print("2. Create account and generate API key")
        print("3. Create .env file with: GROQ_API_KEY=your_key_here")
        return None
    
    return Groq(api_key=api_key)


def generate_solution(client, question, topic):
    """
    Generate pedagogically structured solution using LLM.
    
    Constructs a carefully designed prompt that guides the LLM to produce
    educational content suitable for high school students. The prompt
    emphasizes conceptual understanding, clear step-by-step reasoning,
    and encouraging language.
    
    Solution Structure:
        1. Concept explanation
        2. Step-by-step procedure with reasoning
        3. Clear final answer
        4. Student-appropriate language
        
    Args:
        client (Groq): Authenticated API client
        question (str): Mathematics question text
        topic (str): Subject category (Algebra, Calculus, etc.)
        
    Returns:
        str: Generated solution text
        None: If API call fails
        
    Model Configuration:
        - Model: llama-3.3-70b-versatile (70B parameter model)
        - Temperature: 0.7 (balanced creativity and consistency)
        - Max Tokens: 1000 (sufficient for detailed explanations)
    """
    
    # Construct system prompt to establish AI persona
    system_prompt = (
        "You are an expert mathematics tutor who specializes in making "
        "complex concepts accessible to high school students. Your "
        "explanations are clear, encouraging, and pedagogically sound."
    )
    
    # Construct user prompt with specific requirements
    user_prompt = f"""A student is working on a {topic} problem and needs help.

Question: {question}

Please provide a comprehensive solution that includes:

1. Brief Concept Review: What principle or concept applies here
2. Solution Steps: Each step clearly labeled and explained
3. Step Reasoning: Why each step is necessary
4. Final Answer: Clear conclusion with appropriate units/format
5. Language: Use terminology appropriate for high school level

Structure your response to maximize student understanding and confidence."""

    try:
        # Execute API call to Groq service
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,        # Balanced between consistency and creativity
            max_tokens=1000         # Sufficient length for detailed explanations
        )
        
        # Extract generated solution from response
        solution = completion.choices[0].message.content
        return solution
        
    except Exception as e:
        print(f"Error during solution generation: {e}")
        return None


def load_sample_questions(data_folder, num_samples=5):
    """
    Load representative sample questions from dataset for solution generation.
    
    Implements balanced sampling across topics to ensure diversity in the
    generated solutions. This demonstrates capability across all mathematical
    domains without incurring excessive API costs.
    
    Sampling Strategy:
        - Distribute samples evenly across available topics
        - Select first N questions from each topic directory
        - Stop when target sample count is reached
        
    Args:
        data_folder (str): Root directory containing topic subdirectories
        num_samples (int): Target number of samples to extract
        
    Returns:
        list: Sample dictionaries containing:
            - question (str): Question text
            - topic (str): Subject category
            - file_name (str): Source file identifier
            
    Note:
        For demonstration purposes, 5-10 samples is typically sufficient.
        Generating solutions for entire datasets may be cost-prohibitive
        depending on API pricing and quotas.
    """
    samples = []
    data_path = Path(data_folder)
    
    # Validate directory existence
    if not data_path.exists():
        print(f"Error: Directory '{data_folder}' not found")
        return []
    
    # Identify topic subdirectories
    topic_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not topic_folders:
        print(f"Error: No topic subdirectories found in '{data_folder}'")
        return []
    
    print(f"Sampling questions from '{data_folder}'")
    
    # Calculate samples per topic for balanced representation
    samples_per_topic = max(1, num_samples // len(topic_folders))
    
    # Collect samples from each topic
    for folder in sorted(topic_folders):
        topic = folder.name
        json_files = list(folder.glob("*.json"))[:samples_per_topic]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                question = data.get('problem') or data.get('question') or ''
                
                if question:
                    samples.append({
                        'question': question,
                        'topic': topic,
                        'file_name': json_file.stem
                    })
                    
            except Exception as e:
                print(f"Warning: Error reading {json_file.name}: {e}")
        
        # Stop if target sample count reached
        if len(samples) >= num_samples:
            break
    
    print(f"Loaded {len(samples)} sample questions for processing")
    return samples[:num_samples]


def process_sample_questions(data_folder, num_samples=5):
    """
    Orchestrate end-to-end solution generation pipeline.
    
    Pipeline Stages:
        1. API client initialization
        2. Sample question loading
        3. Iterative solution generation
        4. Result aggregation
        5. Multi-format persistence
        
    Implements rate limiting between requests to respect API guidelines
    and prevent throttling.
    
    Args:
        data_folder (str): Source directory for question samples
        num_samples (int): Number of solutions to generate
        
    Note:
        Each API call consumes quota/credits. Adjust num_samples based
        on available resources and demonstration requirements.
    """
    print("="*70)
    print("AI SOLUTION GENERATION SYSTEM")
    print("="*70)
    
    # Initialize API connection
    client = setup_groq()
    if not client:
        print("\nError: Cannot proceed without valid API credentials")
        return
    
    # Load sample questions
    samples = load_sample_questions(data_folder, num_samples)
    
    if not samples:
        print("Error: No questions available for processing")
        return
    
    print(f"\nProcessing {len(samples)} questions")
    
    results = []
    
    # Generate solution for each sample
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*70}")
        print(f"Processing Question {i}/{len(samples)}")
        print(f"{'='*70}")
        
        question = sample['question']
        topic = sample['topic']
        
        print(f"\nTopic: {topic}")
        print(f"Question: {question}")
        print("\nGenerating solution...")
        
        # Call LLM for solution generation
        solution = generate_solution(client, question, topic)
        
        if solution:
            print(f"\nSolution generated successfully")
            print(f"\n{solution}")
            
            # Aggregate result
            results.append({
                'file_name': sample['file_name'],
                'question': question,
                'topic': topic,
                'solution': solution
            })
        else:
            print(f"\nWarning: Solution generation failed for this question")
        
        # Rate limiting: pause between requests to avoid API throttling
        if i < len(samples):
            time.sleep(1)
    
    # Persist results to disk
    if results:
        save_solutions(results)
    
    print(f"\n{'='*70}")
    print(f"Pipeline Complete: {len(results)}/{len(samples)} solutions generated")
    print(f"{'='*70}")


def save_solutions(results):
    """
    Persist generated solutions in multiple formats for different use cases.
    
    Output Formats:
        1. JSON: Machine-readable structured data for programmatic access
        2. TXT: Human-readable formatted text for manual review
        
    The JSON format enables integration with other systems or APIs, while
    the text format facilitates quick review and quality assessment.
    
    Args:
        results (list): Collection of solution dictionaries
        
    Output Files:
        - outputs/generated_solutions.json: Structured data
        - outputs/solutions_readable.txt: Formatted text
    """
    print(f"\n{'='*70}")
    print("PERSISTING RESULTS")
    print(f"{'='*70}")
    
    # Ensure output directory exists
    Path('outputs').mkdir(exist_ok=True)
    
    # Save as structured JSON
    json_file = 'outputs/generated_solutions.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nStructured data saved: {json_file} ({len(results)} solutions)")
    
    # Generate human-readable text version
    text_file = 'outputs/solutions_readable.txt'
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("AI-GENERATED MATHEMATICS SOLUTIONS\n")
        f.write("Model: Llama 3.3 (70B) via Groq API\n")
        f.write("="*70 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{'='*70}\n")
            f.write(f"SOLUTION {i}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Source File: {result['file_name']}\n")
            f.write(f"Topic: {result['topic']}\n\n")
            f.write(f"Question:\n{result['question']}\n\n")
            f.write(f"Generated Solution:\n{result['solution']}\n\n")
            f.write("\n\n")
    
    print(f"Formatted text saved: {text_file}")
    
    print(f"\nGenerated artifacts ready for:")
    print(f"  - Project submission ({json_file})")
    print(f"  - Quality review ({text_file})")


def main():
    """
    Entry point for solution generation system.
    
    Configuration parameters are centralized here for easy adjustment.
    Modify these values based on your specific requirements and API quotas.
    """
    # System configuration
    data_folder = "dataset/train"     # Source directory for questions
    num_samples = 5                   # Number of solutions to generate
    
    # Execute generation pipeline
    process_sample_questions(data_folder, num_samples)
    
    print("\nExecution complete. Review outputs/ directory for results.")


if __name__ == "__main__":
    main()
