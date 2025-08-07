"""
Prompt templates for the Agentic AI Assistant.
Contains all LLM prompts organized by functionality.
"""

from typing import Dict, Any


def get_router_prompt(user_prompt: str, doc_flag: str, schema: str, qs_examples: str, documentation: str) -> str:
    """
    Generate the router prompt for determining the appropriate route.
    
    Args:
        user_prompt: The user's input query
        doc_flag: Whether a document is uploaded ("yes"/"no")
        schema: Database schema description
        qs_examples: Example question-SQL pairs
        documentation: Database documentation
        
    Returns:
        Formatted router prompt
    """
    return f"""
    You are an intelligent routing agent. Your job is to:
    1. Choose one of the paths: "sql", "search", "document", "comp", "faissdb" based on the user prompt.
    2. Choose:
    - "sql" if the user is asking a question about structured insurance data (e.g. claims, premiums, reserves, IBNR, trends, comparisons across years or products) or something that can be answered from the following database schema:
        {schema}
    - Use this additional documentation to better understand column meanings:
      {documentation}
    - Additionally, here are some examples of SQL-style questions and their corresponding queries (QSPairs):
      {qs_examples}
    -EVEN IF the user also says things like "plot", "draw", "visualize", "graph", "bar chart", etc. — that still means they want structured data **along with** a chart. SO route it to SQL
        Example: "Show me IBNR over years and plot a bar chart" → route = "sql"
    -Route it to "sql" if queries includes the below mentioned:
        - Asks for trends, breakdowns, or aggregations of internal metrics (e.g., IBNR, reserves, severity, premiums, earned/ultimate loss)
        - Ask for trends **within internal data only**
        - Compares **internal data over time or segments** (e.g., years, lines of business, regions)
        - Ask for charts or visualizations ("plot", "bar chart", etc.)
        - Does NOT involve external benchmarking
        Even if the prompt includes words like "compare" or "change", still route to SQL if the context is strictly internal.
    -If the route is "sql", include vanna_prompt, but don't include fuzzy_prompt
        -(eg: User Prompt is "Show me exposure year wise incurred loss and plot a graph", then 
        -vanna_prompt will be "Shoe me exposure year wise incurred loss".
        -Your work is to remove the noise and focus only on things that are required to generate sql query from vanna. SO remove all the extra stuffs out of the user prompt.


    3. "document" ONLY if a document is uploaded (Document Uploaded = yes) AND the question involves updating/reading a document.
    -If the route is "document", include:
        - "vanna_prompt": an SQL-style question to query structured data.
        - "fuzzy_prompt": a natural language description of the header or table to update.

    
    4. Choose "search" if:
        - The user is asking about general or external information
        - Involves real-time info, news, global economic trends, regulations
        - The query cannot be answered by internal structured data or uploaded document
    - If the route is "search", DO NOT include vanna_prompt or fuzzy_prompt.


    5.Choose "comp" when the user is comparing internal data against external data, competitors, or industry benchmarks.
        Examples include peer review, benchmarking, market positioning, or competitive ratios.

        Trigger words/phrases (especially relevant for Actuarial & Finance users):
        - "industry benchmark"
        - "market average"
        - "how do we compare to..."
        - "peer comparison"
        - "market trend vs ours"
        - "against competitors"
        - "vs industry"
        - "benchmarking analysis"
        - "loss ratio gap with peers"
        - "pricing differential with market"
        - "expense ratio compared to competition"
        - "where do we stand in market"
        - "relative to industry"
        - "competitive advantage in reserves"
        - "our severity vs others"
        - "compare to S&P average" / "AM Best stats" / "regulatory benchmark"
    -(e.g.,User Prompt is "Compare IBNR trends with industry benchmarks for exposure year 2025 ")
    - Return Vanna_prompt as well as "Show IBNR trends for exposure year 2025"
    -Do not include fuzzy_prompt
    -Only include relevant columns in vanna_prompt. Do not include ClaimNumber or ID columns unless the user specifically asks for them.

   
    6. Choose "faissdb" when:
    - The prompt asks about the Sparta platform, Earmark Template, Branch Adjustment Template/Module, Projects in Sparta, or any internal process or documentation.
    - The user seems to be referring to internal workflows, or knowledge base content.
    -Example prompts that should be routed to `"faissdb"`:
        - "What are the steps in the Branch Adjustment Module?"
        - "Explain how Earmark Template is used in our process."
        - "Can you summarize Projects in Sparta?"


    Return output strictly in valid JSON format using double quotes and commas properly.
    DO NOT include any trailing commas. Your JSON must be parseable by Python's json.loads().

    Examples:

    For SQL:
    {{
        "route": "sql",
        "vanna_prompt": "Show IBNR trends for exposure year 2025"
    }}

    For Document:
    {{
        "route": "document",
        "vanna_prompt": "SELECT policy_id, total_loss FROM policies WHERE year = 2024",
        "fuzzy_prompt": "Update the table under 'Loss Overview' for 2024"
    }}

    For Comp:
    {{
         "route": "comp",
         "vanna_prompt": "Show IBNR trends for exposure year 2025"
    }}

    For Search:
    {{
        "route": "search"
    }}

    For faissdb 
    {{
    "route": "faissdb"
    }}

    User Prompt: "{user_prompt}"
    Document Uploaded: {doc_flag}
    """


def get_chart_suggestion_prompt(sample_data: str) -> str:
    """
    Generate prompt for suggesting chart types based on data.
    
    Args:
        sample_data: JSON string of sample DataFrame data
        
    Returns:
        Chart suggestion prompt
    """
    return f"""
    You are a data visualization assistant.

    Here is the top of a pandas DataFrame:
    {sample_data}

    Your task:
    - Identify a good chart (bar, line, or pie) that best represents this data.
    - Choose 1 column for the x-axis (categorical or time-based), and 1 or more numeric columns for the y-axis.
    - If multiple y columns are appropriate (e.g. IBNR, IncurredLoss), return them as a list.

    Return your answer in JSON like:
    {{ "type": "bar", "x": "ExposureYear", "y": ["IncurredLoss", "IBNR"] }}

    If no chart is suitable, return: "none"
    """


def get_serp_general_summary_prompt(combined_text: str, user_prompt: str, sql_snippet: str = "") -> str:
    """
    Generate prompt for creating general summary from search results.
    
    Args:
        combined_text: Combined text from search snippets
        user_prompt: Original user prompt
        sql_snippet: Optional SQL context for comparison
        
    Returns:
        General summary prompt
    """
    if sql_snippet:
        return f"""
        You are an insurance and actuarial analyst comparing internal company data with external web results.

        Use the following INTERNAL SQL DATA ONLY FOR CONTEXT. **Do not include internal tables or numbers in your output.**

        🧾 Internal SQL Query:
        {sql_snippet}

        ---

        Now, using only the following external web snippets, write a summary:

        🔍 Web Snippets:
        {combined_text}

        ---

        User Prompt:
        "{user_prompt}"


        🔽 Your Task:
        - Summarize **only what is found in the external data**
        - DO NOT display the internal SQL data or repeat it
        - Be concise, no more than **6–8 lines**
        - Include **percentages, currency, loss ratios, IBNR**, and other KPIs found in the web
        - Avoid repeating full articles or sentences

        Output format:
        1. 📌 Start with a summary of overall findings.
        2. 🔢 Then list 3–4 **quantitative highlights**.
        3. 💬 End with any notable quote or number from a source if applicable.
        4. Can include a table with numerical insights as well, but not the internal data or tabular data. Only if you found it in external data.
        """
    else:
        return f"""
        You are an insurance and actuarial analyst.

        Your task is to extract **concise and numerically rich insights** from the following web snippets, in response to this user query:

        "{user_prompt}"

        Snippets:
        {combined_text}

        Your summary should:
        - Be structured and no more than **6–8 lines**
        - Include **percentages**, **currency values**, **ratios**, **dates**, and **growth trends**
        - Mention key **KPIs** (e.g., IBNR, premiums, loss ratios, reserves)
        - Avoid repeating the snippets. Instead, **synthesize them**
        - If no numbers are found, say so explicitly

        Output format:
        1. 📌 Start with a summary of overall findings.
        2. 🔢 Then list 3–4 **quantitative highlights**.
        3. 💬 End with any notable quote or number from a source if applicable.
        4. Can include a table with numerical insights as well
        """


def get_comparison_summary_prompt(sql_df: str, web_links: str, external_summary: str) -> str:
    """
    Generate prompt for creating comparison summary.
    
    Args:
        sql_df: SQL DataFrame in markdown format
        web_links: Web links and summaries
        external_summary: External summary
        
    Returns:
        Comparison summary prompt
    """
    return f"""
    You are an actuarial analyst comparing internal structured data with external insurance insights.

    Your job is to:
    1. Analyze differences, similarities, and gaps between internal company data and external web sources.
    2. Focus heavily on **numerical metrics** such as:
    - IBNR, Incurred Loss, Ultimate Loss
    - Premiums, Loss Ratios
    - Exposure Years, Percent changes

    3. Highlight:
    - Trends (increases/decreases)
    - Matching vs. diverging figures
    - Numerical differences or % differences

    🧾 Internal SQL Output (Top 5 rows, tabular format):
    {sql_df}

    🌐 External Web Insights:
    {web_links}

    💬 General Summary:
    {external_summary}

    Return your final answer as a **clearly structured comparison**.
    Prefer a short table or bullet points with side-by-side numbers wherever appropriate.
    Start with a one-liner summary, then details.
    """


def get_faiss_summary_prompt(user_prompt: str, content_snippets: str) -> str:
    """
    Generate prompt for FAISS knowledge base summary.
    
    Args:
        user_prompt: User's original query
        content_snippets: Retrieved document content
        
    Returns:
        FAISS summary prompt
    """
    return f"""
    Based on the following retrieved document chunks from internal knowledge base, answer the user's query:

    User Prompt: {user_prompt}

    Documents:
    {content_snippets}

    Provide a concise and structured answer with key points or numeric details if found.
    """


def get_document_table_prompt(structure_string: str, fuzzy_prompt: str) -> str:
    """
    Generate prompt for identifying correct table in document.
    
    Args:
        structure_string: Document structure description
        fuzzy_prompt: User's fuzzy prompt for table identification
        
    Returns:
        Document table identification prompt
    """
    return f"""
        You are helping identify the correct table to update in a Word document.
        Each table has: index, rows x cols, and list of column headers.

        Document structure:
        {structure_string}

        Instruction:
        \"\"\"{fuzzy_prompt}\"\"\"

        Return strictly in JSON:
        {{ "header_text": "...", "table_index_under_header": 0 }}
    """


def get_follow_up_questions_prompt(user_prompt: str) -> str:
    """
    Generate prompt for suggesting follow-up questions.
    
    Args:
        user_prompt: Original user query
        
    Returns:
        Follow-up questions prompt
    """
    return f"""
    Based on the following insurance-related user query:
    "{user_prompt}"

    Suggest 3 intelligent follow-up questions the user could ask next. Keep them short, relevant, and not repetitive.
    Return them as a plain list.
    """


def get_title_generation_prompt(prompt: str) -> str:
    """
    Generate prompt for creating chat titles.
    
    Args:
        prompt: User's original prompt
        
    Returns:
        Title generation prompt
    """
    return f"Summarize the following user query into a short title:\n\n'{prompt}'\n\nKeep it under 7 words."


# Prompt templates for different scenarios
PROMPT_TEMPLATES = {
    "router": get_router_prompt,
    "chart_suggestion": get_chart_suggestion_prompt,
    "serp_general_summary": get_serp_general_summary_prompt,
    "comparison_summary": get_comparison_summary_prompt,
    "faiss_summary": get_faiss_summary_prompt,
    "document_table": get_document_table_prompt,
    "follow_up_questions": get_follow_up_questions_prompt,
    "title_generation": get_title_generation_prompt
} 