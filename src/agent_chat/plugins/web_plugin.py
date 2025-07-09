import hashlib
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from markdownify import markdownify as md
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Suppress the oauth2client file_cache warning
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

from googleapiclient.discovery import build

from agent_chat.tool_registry import ToolCall


class WebPlugin:
    """Plugin for fetching web content and reading cached files."""

    def __init__(self, cache_timeout: int = 3600):  # 1 hour
        self.cache_timeout = cache_timeout
        self.cache_dir = self._get_cache_dir()
        self.search_results = {}  # {link_id: {'url': ..., 'title': ...}}
        self.next_link_id = 1

    def _process_url_for_display(self, url: str, max_length: int = 80) -> str:
        """Process URL for display: decode and truncate if needed."""
        # Decode URL-encoded characters
        decoded_url = unquote(url)

        # If short enough, return as-is
        if len(decoded_url) <= max_length:
            return decoded_url

        # Parse URL for smart truncation
        parsed = urlparse(decoded_url)
        domain = parsed.netloc
        path_parts = parsed.path.strip("/").split("/")

        # Always keep domain
        base = domain

        # If path is short, keep it all
        if len(decoded_url) <= max_length * 1.5:
            # Just ellipsis in the middle
            keep_start = max_length // 2 - 3
            keep_end = max_length // 2 - 3
            return decoded_url[:keep_start] + "..." + decoded_url[-keep_end:]

        # For very long URLs, show domain + first path segment + end
        if path_parts:
            first_segment = path_parts[0][:20] if path_parts[0] else ""
            if first_segment:
                base = f"{domain}/{first_segment}"

            # Add some context from the end if there's room
            remaining = max_length - len(base) - 10  # Reserve space for .../...
            if remaining > 10 and len(path_parts) > 1:
                last_part = path_parts[-1][:remaining]
                return f"{base}/.../...{last_part}"
            else:
                return f"{base}/..."

        return f"{domain}/..."

    def _get_cache_dir(self) -> Path:
        """Get XDG cache directory for web fetch."""
        xdg_cache = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = Path(xdg_cache) / "agent_chat_web_fetch"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _normalize_url(self, url: str) -> str:
        """Normalize URL and upgrade to HTTPS."""
        if url.startswith("http://"):
            url = url.replace("http://", "https://", 1)
        return url

    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path for URL."""
        parsed = urlparse(url)
        domain = parsed.netloc

        # Create hash of full URL for filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

        # Extract page name from path for readability
        path_parts = parsed.path.strip("/").split("/")
        page_name = path_parts[-1] if path_parts and path_parts[-1] else "index"
        if "." in page_name:
            page_name = page_name.split(".")[0]

        filename = f"{page_name}_{url_hash}.md"

        domain_dir = self.cache_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        return domain_dir / filename

    def _get_relative_path(self, cache_path: Path) -> str:
        """Get relative path from cache directory for display to model."""
        try:
            # Get relative path from cache directory
            relative = cache_path.relative_to(self.cache_dir)

            # Clean up the filename - remove hash suffix for display
            parts = list(relative.parts)
            if parts:
                filename = parts[-1]
                if "_" in filename and filename.endswith(".md"):
                    # Remove hash suffix (e.g., "news_19b903d5.md" -> "news.md")
                    base_name = filename.split("_")[0]
                    parts[-1] = f"{base_name}.md"

            return str(Path(*parts))
        except ValueError:
            # Fallback if path is not relative to cache dir
            return str(cache_path)

    def _resolve_relative_path(self, relative_path: str) -> Path:
        """Resolve relative path to full cache path, handling cleaned filenames."""
        # Start with the relative path
        test_path = self.cache_dir / relative_path

        # If the exact path exists, return it
        if test_path.exists():
            return test_path

        # Otherwise, we need to find the actual file with hash suffix
        # Extract directory and base filename
        relative_parts = Path(relative_path).parts
        if len(relative_parts) >= 2:
            domain = relative_parts[0]
            filename = relative_parts[-1]

            if filename.endswith(".md"):
                base_name = filename[:-3]  # Remove .md
                domain_dir = self.cache_dir / domain

                # Look for files matching the pattern base_name_*.md
                if domain_dir.exists():
                    for file_path in domain_dir.glob(f"{base_name}_*.md"):
                        return file_path

        # If we can't find it, return the original path (will fail later with proper error)
        return test_path

    def _is_cache_valid(self, cache_path: Path, max_age: int = None) -> bool:
        """Check if cached file is still valid."""
        if not cache_path.exists():
            return False

        file_time = cache_path.stat().st_mtime
        timeout = max_age if max_age is not None else self.cache_timeout
        return time.time() - file_time < timeout

    def _fetch_with_wget(self, url: str) -> str:
        """Fetch content using wget for better bot detection evasion."""
        try:
            # Create temporary file for wget output
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".html", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name

            # Use wget with Firefox user agent
            cmd = [
                "wget",
                "-O",
                tmp_path,
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64; rv:119.0) Gecko/20100101 Firefox/119.0",
                "--timeout=30",
                "--tries=3",
                url,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                raise RuntimeError(f"wget failed: {result.stderr}")

            # Read the downloaded content
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Clean up temp file
            os.unlink(tmp_path)

            # Convert HTML to markdown
            markdown_content = md(html_content, heading_style="ATX")
            return markdown_content

        except Exception as e:
            raise RuntimeError(f"wget failed: {str(e)}")

    def _fetch_with_jina(self, url: str) -> str:
        """Fetch content using Jina AI r.jina.ai as fallback."""
        try:
            import urllib.error
            import urllib.request

            # Use Jina AI reader endpoint
            jina_url = f"https://r.jina.ai/{url}"

            # Create request with user agent
            req = urllib.request.Request(
                jina_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:119.0) Gecko/20100101 Firefox/119.0"
                },
            )

            # Fetch content with timeout
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode("utf-8")

            return content

        except Exception as e:
            raise RuntimeError(f"Jina AI fetch failed: {str(e)}")

    def _fetch_content(self, url: str) -> str:
        """Fetch content with wget fallback to Jina AI."""
        try:
            # Try wget first
            logger.debug(f"Attempting to fetch with wget: {url}")
            return self._fetch_with_wget(url)
        except RuntimeError as e:
            # If wget fails, try Jina AI as fallback
            logger.debug(f"wget failed, trying Jina AI fallback: {str(e)}")
            try:
                return self._fetch_with_jina(url)
            except RuntimeError as jina_error:
                # Both methods failed
                raise RuntimeError(
                    f"Both wget and Jina AI failed. wget: {str(e)}, Jina AI: {str(jina_error)}"
                )

    def _save_to_cache(self, cache_path: Path, content: str) -> None:
        """Save content to cache file."""
        try:
            cache_path.write_text(content, encoding="utf-8")
            logger.debug(f"Cached content to {cache_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving to cache {cache_path}: {str(e)}")

    def _get_cached_content(self, cache_path: Path) -> str:
        """Read cached content from file."""
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Error reading cache file {cache_path}: {str(e)}")

    def _resolve_source_to_url(self, source):
        """Resolve source (link_id:N or URL) to URL and source_info.

        Returns:
            tuple: (url, source_info) on success, (None, error_message) on error
        """
        # Parse source format: either "link_id:N" or direct URL
        if isinstance(source, str) and source.startswith("link_id:"):
            try:
                link_id = int(source[8:])  # Remove "link_id:" prefix
            except ValueError:
                return (
                    None,
                    f"Error: Invalid link ID format. Use 'link_id:1', 'link_id:2', etc. Got: '{source}'",
                )

            if link_id < 1:
                return None, f"Error: Link ID must be positive. Got: link_id:{link_id}"

            if link_id not in self.search_results:
                available = [f"link_id:{id}" for id in self.search_results.keys()]
                return (
                    None,
                    f"Error: Link ID {link_id} not found. Available: {available}. Run web_search first to get links.",
                )

            url = self.search_results[link_id]["url"]
            source_info = f"Link [{link_id}]: {self.search_results[link_id]['title']}"
            logger.info(f"Resolved link_id:{link_id} to URL: {url}")
            return url, source_info

        elif isinstance(source, str):
            # Validate URL using urllib
            try:
                parsed = urlparse(source.strip())
                if not all([parsed.scheme, parsed.netloc]):
                    return (
                        None,
                        f"Error: Invalid URL format. Must include scheme and domain. Got: '{source}'",
                    )
                if parsed.scheme not in ("http", "https"):
                    return (
                        None,
                        f"Error: URL must use http or https scheme. Got: '{parsed.scheme}'",
                    )
                url = source.strip()
                source_info = url
                return url, source_info
            except Exception:
                return None, f"Error: Invalid URL format. Got: '{source}'"

        else:
            return (
                None,
                f"Error: Invalid source format. Use 'link_id:1' for search results or 'https://...' for direct URLs. Got: '{source}'",
            )

    async def _analyze_with_ai(self, content: str, url: str, prompt: str = None) -> str:
        """Analyze content with OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "AI analysis not available - OPENAI_API_KEY not set"

        client = AsyncOpenAI(api_key=api_key)

        system_prompt = f"""You are analyzing web content that has been converted from HTML to markdown.

The content from {url} is:

{content}

Please analyze and summarize the main content of this webpage."""

        # Use custom prompt if provided, otherwise default instruction
        user_prompt = (
            prompt
            if prompt
            else "Analyze and summarize the main content of this webpage."
        )

        try:
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
                max_tokens=2048,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"AI analysis failed: {str(e)}"

    def web_search(self, query: str) -> str:
        """Perform web searches to retrieve current information. Results can be fetched using their link IDs."""

        api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
        if not api_key:
            return "Error: GOOGLE_SEARCH_API_KEY environment variable not set"

        cx = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
        if not cx:
            return "Error: GOOGLE_SEARCH_ENGINE_ID environment variable not set"

        # Keep previous search results and continue link ID sequence

        try:
            service = build("customsearch", "v1", developerKey=api_key)
            resp = service.cse().list(q=query, cx=cx, num=10).execute()

            results = []
            for item in resp.get("items", []):
                title = item.get("title", "No title")
                url = item.get("link", "No link")
                description = item.get("snippet", "No description")

                # Store the result with link ID
                link_id = self.next_link_id
                self.search_results[link_id] = {
                    "url": url,
                    "title": title,
                    "description": description,
                }
                self.next_link_id += 1

                # Format the result with processed URL
                display_url = self._process_url_for_display(url)
                results.append(f"[{link_id}] {title}")
                results.append(f"    {display_url}")
                results.append(f"    {description}")
                results.append("")  # Empty line for spacing

            if not results:
                return f"No search results found for query: {query}"

            header = f"Search results for '{query}':\n\n"

            return header + "\n".join(results)

        except Exception as e:
            return f"Error performing web search: {str(e)}"

    async def web_query_page(
        self, source, prompt: str = None, max_age: int = None
    ) -> str:
        """Query and analyze web content using AI from a URL or search result link reference. Returns AI-generated summary and caches content for later reading.

        Args:
            source: Either 'link_id:N' for search results or 'https://...' for direct URLs
            prompt: Optional natural language instruction for what to analyze in the content
            max_age: Optional cache age in seconds (0 for always fresh, default is 3600 seconds)
        """
        try:
            # Resolve source to URL and source info
            url, source_info = self._resolve_source_to_url(source)
            if url is None:
                return source_info  # source_info contains error message

            # Normalize URL
            normalized_url = self._normalize_url(url)
            cache_path = self._get_cache_path(normalized_url)

            # Check cache
            cache_hit = self._is_cache_valid(cache_path, max_age)

            if cache_hit:
                logger.debug(f"Using cached content from {cache_path}")
                content = self._get_cached_content(cache_path)
            else:
                logger.debug(f"Fetching content from {normalized_url}")
                content = self._fetch_content(normalized_url)
                self._save_to_cache(cache_path, content)

            # Analyze with AI
            logger.debug("Analyzing content with AI")
            summary = await self._analyze_with_ai(content, normalized_url, prompt)

            # Return simple format with source attribution
            return f"Analysis of {source_info}:\n\n{summary}"

        except Exception as e:
            return f"Error analyzing page: {str(e)}"

    async def web_read_page(self, source, max_age: int = None) -> str:
        """Read the full content of a web page from a URL or search result link reference. Returns raw content without AI analysis.

        Args:
            source: Either 'link_id:N' for search results or 'https://...' for direct URLs
            max_age: Optional cache age in seconds (0 for always fresh, default is 3600 seconds)
        """
        try:
            # Resolve source to URL and source info
            url, source_info = self._resolve_source_to_url(source)
            if url is None:
                return source_info  # source_info contains error message

            # Normalize URL and get cache path
            normalized_url = self._normalize_url(url)
            cache_path = self._get_cache_path(normalized_url)

            # Check cache first
            cache_hit = self._is_cache_valid(cache_path, max_age)

            if cache_hit:
                logger.debug(f"Reading cached content from {cache_path}")
                content = self._get_cached_content(cache_path)
            else:
                # Fetch fresh content
                logger.debug(f"Fetching fresh content from {normalized_url}")
                content = self._fetch_content(normalized_url)
                self._save_to_cache(cache_path, content)

            return f"Content from {source_info}:\n\n{content}"

        except Exception as e:
            return f"Error reading page: {str(e)}"

    def hook_provide_tools(self):
        """Return tools this plugin provides for auto-registration."""
        return [self.web_search, self.web_query_page, self.web_read_page]

    def hook_provide_system_prompt(self):
        """Return system prompt addition for web functionality."""
        return """
## Web Search and Content Tools

Available web tools:
- **web_search(query)**: Find current information online, returns numbered links [1], [2], etc.
- **web_query_page(source, prompt=None, max_age=None)**: AI-powered analysis of web content with natural language prompts
- **web_read_page(source, max_age=None)**: Read full raw content of web page (no AI processing)

**Robust fetching**: Uses wget first, then automatically falls back to Jina AI (r.jina.ai) for PDFs and sites that block wget.

**Search Operators**: Supports standard Google search operators like `site:`, `filetype:`, `intitle:`, `"exact phrases"`, `OR`, `-exclude`, etc.

Usage patterns:
- Use web_search() to find information, then reference results using link_id format
- Use web_query_page() with detailed natural language prompts:
  * web_query_page(source="link_id:1", prompt="What are the main arguments presented in this article?")
  * web_query_page(source="link_id:1", prompt="Summarize the pricing information and compare the different plans")
  * web_query_page(source="link_id:1", prompt="What evidence does the author provide to support their claims?")
  * web_query_page(source="link_id:1", prompt="What's new here?", max_age=0) - force fresh analysis
- Use web_read_page() only when you need the complete raw content for detailed analysis:
  * web_read_page(source="link_id:1") - uses default cache (1 hour)
  * web_read_page(source="link_id:1", max_age=0) - always fetch fresh content
- prefer web_query_page() for most use cases, try to be detailed and specific in your prompts.
- web_read_page() may return too much content, so use it sparingly, only on trusted sources.
- Source format: Use "link_id:N" for search results or "https://..." for direct URLs
- Links are numbered for easy reference - you can say "According to link [3]..." in responses
- Content is automatically cached for efficient re-access

Link ID system:
- Link IDs grow continuously across multiple web_search calls (never reset)
- All link IDs remain valid throughout the conversation session
- Use explicit format: "link_id:1" for search results, "https://example.com" for direct URLs

Link references:
- Always refer to search results using link_id:N format (e.g., "See link_id:1 for details")
- These automatically expand to show title and URL to users
""".strip()

    def _expand_link_references(self, text: str) -> str:
        """Helper method to expand link_id:N references to user-friendly format.

        Args:
            text: Input text that may contain link_id:N patterns

        Returns:
            Text with link_id:N expanded to [N] Title: URL format
        """
        import re

        def replace_link_id(match):
            try:
                link_id = int(match.group(1))
                if link_id in self.search_results:
                    result_info = self.search_results[link_id]
                    title = result_info["title"]
                    url = result_info["url"]
                    # Expand to user-friendly format: [ID] Title: URL
                    return f"[{link_id}] {title}: {url}"
                else:
                    # Keep original if link_id not found
                    return match.group(0)
            except ValueError:
                # Keep original if invalid format
                return match.group(0)

        # Replace all link_id:N patterns
        return re.sub(r"link_id:(\d+)", replace_link_id, text)

    async def hook_modify_tool_call(self, tool_call):
        """Enhance tool arguments by expanding link_id references to user-friendly format.

        This hook processes tool arguments before execution to expand link_id:N references
        into more informative text that includes the title and URL from search results.
        """
        # Skip expansion for web plugin's own tools - they need raw link_id format
        web_tools = {"web_search", "web_query_page", "web_read_page"}
        if tool_call.name in web_tools:
            return tool_call

        # Process all string arguments in the tool call
        modified_args = tool_call.arguments.copy()

        for key, value in tool_call.arguments.items():
            if isinstance(value, str):
                modified_args[key] = self._expand_link_references(value)

        # Return a new ToolCall with modified arguments
        return ToolCall(name=tool_call.name, arguments=modified_args)

    async def hook_modify_model_response(self, content: str) -> str:
        """Enhance model responses by expanding link_id references to user-friendly format.

        This hook processes direct model responses to expand link_id:N references
        into more informative text that includes the title and URL from search results.
        """
        return self._expand_link_references(content)
