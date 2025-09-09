from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from config.settings import settings
from typing import Optional
import google.generativeai as genai

class LLMService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            temperature=0.3
        )
    
    def answer_question(self, context: str, question: str) -> str:
        """
        Answer user's question based on research paper content
        
        Args:
            context: Research paper content
            question: User's question
            
        Returns:
            Answer based on the research paper
        """
        system_prompt = """You are an expert research assistant with deep knowledge in analyzing academic papers. Your task is to provide comprehensive, accurate, and well-structured answers based on the provided research paper content.

Instructions for answering:
1. **Thoroughness**: Provide a complete answer that fully addresses the question - don't be overly brief
2. **Structure**: Organize your response with clear paragraphs and logical flow
3. **Evidence-based**: Support your answer with specific information, data, findings, or quotes from the research paper
4. **Context**: Include relevant background information and explain technical concepts when necessary
5. **Accuracy**: Use only information explicitly stated or clearly implied in the provided content
6. **Clarity**: Write in clear, accessible language while maintaining academic rigor
7. **Completeness**: Cover all aspects of the question if multiple parts are involved

Response guidelines:
- Start with a direct answer to the main question
- Provide detailed explanations with supporting evidence from the paper
- Include relevant statistics, methodologies, or findings when applicable
- Quote specific passages when they directly support your answer
- If the information is not available in the paper, clearly state: "The provided research paper does not contain information about [specific aspect]"
- Aim for comprehensive coverage rather than brevity

Research Paper Content:
{context}"""
        
        human_prompt = f"Based on the research paper content provided above, please answer the following question thoroughly and comprehensively using bullet points for clear organization:\n\nQuestion: {question}\n\nProvide a detailed response with bullet points that fully addresses this question using the available research content. Use clear headings and bullet point structure for maximum readability."
        
        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_summary(self, context: str, query: str) -> str:
        """
        Generate summary of research paper based on user's specific query
        
        Args:
            context: Research paper content
            query: User's query for what to summarize
            
        Returns:
            Summary focused on the user's query
        """
        system_prompt = """You are an expert research assistant specializing in academic paper analysis and synthesis. Generate a comprehensive, well-structured summary of the research paper content based on the user's specific focus area.

Instructions:
1. **Analysis**: Thoroughly read and analyze the entire research paper content provided
2. **Relevance**: Extract and prioritize information that directly relates to the user's query/focus area
3. **Structure**: Organize the summary with clear headings and logical progression
4. **Comprehensiveness**: Include all relevant aspects found in the paper, such as:
   - Key concepts, definitions, and theoretical frameworks
   - Research methodologies and experimental approaches
   - Important findings, results, and data points
   - Statistical evidence and quantitative measures
   - Authors' conclusions, implications, and recommendations
   - Limitations, challenges, or gaps identified
   - Future research directions (if mentioned)

5. **Academic Rigor**: Maintain scholarly tone and precision
6. **Clarity**: Use clear headings, bullet points, and numbered lists for optimal readability
7. **Completeness**: If the query topic isn't extensively covered, clearly indicate the scope of available information
8. **Evidence**: Include specific examples, case studies, or data points when relevant

Format your response with:
- Clear section headings
- Logical flow from general concepts to specific findings
- Bullet points for lists of findings or recommendations
- Proper emphasis on key insights and conclusions

Research Paper Content:
{context}"""
        
        human_prompt = f"Generate a detailed, well-structured summary focusing specifically on: **{query}**\n\nPlease provide a comprehensive analysis that covers all relevant aspects of this topic found in the research paper. Structure your response with clear headings and ensure thorough coverage of the available content."
        
        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"