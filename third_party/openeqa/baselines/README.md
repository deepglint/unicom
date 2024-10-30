# OpenEQA Baselines

The commands required to run serveral baselines are listed below. Some baselines are labeled (language-only) because the model only receives an EQA question $Q$ and must answer based on its prior knowledge of the world. Others baselines are vision-language models (VLMs), which are able to jointly process the question $Q$ and image frames from the episode history $H$.

1. GPT-4 (language-only)

   ```bash
   # requires setting the OPENAI_API_KEY environment variable
   python openeqa/baselines/gpt4.py --dry-run  # remove --dry-run to process the full benchmark
   ```

2. LLaMA (language-only)

   First, download LLaMA weights in the Hugging Face format from [here](https://huggingface.co/meta-llama). Then, run:

   ```bash
   python openeqa/baselines/llama.py -m <path/to/hf/weights>
   ```

3. GPT-4V (vision + language)

   ```bash
   # requires setting the OPENAI_API_KEY environment variable
   python openeqa/baselines/gpt4v.py --num-frames 50 --dry-run  # remove --dry-run to process the full benchmark
   ```

4. Gemini Pro (language-only)

   ```bash
   # requires setting the GOOGLE_API_KEY environment variable
   python openeqa/baselines/gemini-pro.py --dry-run  # remove --dry-run to process the full benchmark
   ```

5. Gemini Pro Vision (vision + language)

   ```bash
   # requires setting the GOOGLE_API_KEY environment variable
   python openeqa/baselines/gemini-pro-vision.py --num-frames 15 --dry-run  # remove --dry-run to process the full benchmark
   ```

6. Claude 3 (vision + language)

   ```bash
   # requires setting the ANTHROPIC_API_KEY environment variable
   python openeqa/baselines/claude-vision.py --num-frames 20 --dry-run  # remove --dry-run to process the full benchmark
   ```
