# Text-Based Watch Embedding Generation v2

This system generates semantic embeddings for watches using text descriptions instead of images. It leverages OpenAI's GPT models to create rich descriptions focusing on design language and aesthetic vibe, then creates embeddings using OpenAI's text-embedding-3-small model.

## 🎯 Overview

Instead of using image-based embeddings, this approach:
1. **Generates AI descriptions** of watches focusing on design language and aesthetic vibe
2. **Creates text embeddings** from these descriptions using OpenAI's embedding API
3. **Focuses on vibes and design** rather than technical specifications

## 🔧 Setup

### 1. Install Dependencies
```bash
cd embedding_generation_v2
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key
Create a `.env` file in this directory:
```bash
cp env_example.txt .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Ensure Production Scrape Data Exists
Make sure the production scrape data exists at:
```
../production_scrape_20250601_175426/data/final_scrape.csv
```

This contains the latest scraped watch data with comprehensive specifications.

## 🚀 Usage

Run the embedding generation:
```bash
python generate_text_embeddings.py
```

The script will:
1. Load existing watch data from the backend
2. Generate AI descriptions for each watch focusing on design language
3. Create embeddings from the descriptions
4. Save results to the `output/` directory

## 📋 Generated Description Format

The AI generates descriptions following this prompt:
> "Give me a description for the [Brand] [Model] watch. Focus on the vibe and design language of this timepiece. Describe the overall aesthetic and design philosophy, visual character and personality, mood and feeling it conveys, design elements that make it distinctive, and target audience or lifestyle it represents. Do NOT describe specific technical specifications, strap/band details, exact measurements, price information, or availability. Keep the description concise (2-3 sentences) and focused on visual and emotional aspects."

### Example Generated Description
**Rolex Submariner**: "The Rolex Submariner embodies a bold, utilitarian aesthetic with its clean, geometric lines and robust presence that exudes confidence and reliability. Its design philosophy balances professional functionality with timeless elegance, creating a visual language that speaks to adventurous spirits and those who value precision craftsmanship, while maintaining an understated luxury that works equally well in boardrooms and underwater expeditions."

## 📁 Output Files

The script generates several output files in the `output/` directory:

- **`text_embeddings.npy`**: NumPy array of embedding vectors
- **`enhanced_watch_metadata.pkl`**: Complete watch data with AI descriptions and embeddings
- **`enhanced_watch_metadata.json`**: Human-readable metadata (without embeddings)
- **`generation_stats.json`**: Statistics about the generation process
- **Intermediate files**: Saved every 10 watches to prevent data loss

## 🔄 Integration with Backend

**✅ Full Compatibility**: The text-based embeddings are saved in the **exact same format** as the image-based embeddings, making them drop-in replacements.

### File Format Compatibility
- **`watch_embeddings.pkl`**: NumPy array of embedding vectors (same as image system)
- **`watch_metadata.pkl`**: List of watch dictionaries with `index` field (same as image system)
- **Same dimensions**: Backend expects and gets the embedding matrix in identical format

### Automatic Integration
The script automatically saves files to both locations:
1. **`./output/`**: For backup and inspection
2. **`../backend/data/`**: Direct backend integration (production files)

### Manual Integration (if needed)
If you prefer manual file replacement:
```bash
cp output/watch_embeddings.pkl ../backend/data/watch_embeddings.pkl
cp output/watch_metadata.pkl ../backend/data/watch_metadata.pkl
```

### Backend Changes Required
**None!** The backend will automatically work with text embeddings because:
- File names are identical
- Data structures are identical  
- The only difference is the embedding source (text vs image)

## ⚙️ Configuration

### Models Used
- **Description Generation**: `gpt-3.5-turbo`
- **Text Embeddings**: `text-embedding-3-small`

### Rate Limiting
- Description requests: 1 second delay
- Embedding requests: 0.1 second delay

### Customization
You can modify the prompt in the `generate_watch_description()` method to focus on different aspects of the watches.

## 💰 Cost Estimation

For 3,963 watches (latest production scrape):
- **GPT-3.5-turbo**: ~$4.00 (description generation)
- **text-embedding-3-small**: ~$0.10 (embedding generation)
- **Total**: ~$4.10

## 🔍 Key Features

- **Robust error handling** with fallback descriptions
- **Intermediate saving** every 10 watches to prevent data loss
- **Rate limiting** to respect OpenAI API limits
- **Multiple output formats** for flexibility
- **Progress tracking** with detailed logging
- **Environment-based configuration** for security

## 🎨 Design Focus

The generated descriptions emphasize:
- **Visual aesthetics** and design philosophy
- **Emotional character** and personality
- **Target lifestyle** and audience
- **Distinctive design elements**
- **Overall vibe** and mood

**Specifically excludes**:
- Technical specifications
- Strap/band details
- Exact measurements
- Price information
- Availability status

## 🚨 Important Notes

1. **API Key Security**: Never commit your `.env` file to version control
2. **Rate Limits**: The script includes delays to respect OpenAI's rate limits
3. **Cost Monitoring**: Monitor your OpenAI usage during generation
4. **Backup**: Intermediate files are saved to prevent data loss
5. **Quality**: Generated descriptions focus on aesthetic and emotional aspects

## 🔄 Troubleshooting

**API Key Error**: Ensure your `.env` file exists and contains a valid OpenAI API key

**Rate Limit Errors**: The script includes built-in delays, but you may need to increase them

**Memory Issues**: For large datasets, consider processing in smaller batches

**Network Issues**: The script will continue from where it left off using intermediate saves 