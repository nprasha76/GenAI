
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def main(url:str)->str:
    # Configure the crawler with default markdown generation
    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)

        if result.success:
            print("Raw Markdown Output:\n")
            print(result.markdown)  # The unfiltered markdown from the page
        else:
            print("Crawl failed:", result.error_message)
    return result.markdown if result.success else ""        

def downloadcontent_withmarkdown(url):
    
    content=asyncio.run(main(url))
    # Initialize the crawler
    
    return content