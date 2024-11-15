import streamlit as st
import spacy
import numpy as np
from langchain_community.llms import OpenAI
from textblob import TextBlob
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List

class PromptQualityTester:
    def __init__(self, api_key: str):
        """Initialize the tester with OpenAI API key"""
        self.llm = OpenAI(api_key=api_key)
        self.nlp = spacy.load('en_core_web_sm')
        self.history = []
        
    def test_prompt(self, prompt: str, expected_pattern: str) -> Dict:
        """Test a prompt and return quality metrics"""
        try:
            # Generate response from LLM
            response = self.llm.generate([prompt]).generations[0][0].text
            
            # Calculate metrics
            metrics = self.evaluate_response(prompt, expected_pattern, response)
            
            # Save result
            result = {
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'expected': expected_pattern,
                'response': response,
                'metrics': metrics
            }
            self.history.append(result)
            return result
            
        except Exception as e:
            st.error(f"Error testing prompt: {str(e)}")
            return None
        
    def evaluate_response(self, prompt: str, expected: str, actual: str) -> Dict:
        """Calculate quality metrics for the response"""
        metrics = {
            'clarity': self._measure_clarity(actual),
            'relevance': self._measure_relevance(expected, actual),
            'completeness': self._measure_completeness(expected, actual),
            'consistency': self._measure_consistency(actual),
            'conciseness': self._measure_conciseness(actual)
        }
        metrics['overall'] = np.mean(list(metrics.values()))
        return metrics

    def _measure_clarity(self, text: str) -> float:
        """Measure text clarity using readability metrics"""
        blob = TextBlob(text)
        sentences = len(blob.sentences)
        words = len(blob.words)
        if sentences == 0:
            return 0.0
        avg_sentence_length = words / sentences
        return min(1.0, 2.0 / (1 + np.exp(avg_sentence_length / 20)))

    def _measure_relevance(self, expected: str, actual: str) -> float:
        """Measure semantic similarity between expected and actual"""
        doc1 = self.nlp(expected)
        doc2 = self.nlp(actual)
        return doc1.similarity(doc2)

    def _measure_completeness(self, expected: str, actual: str) -> float:
        """Measure if all expected elements are present"""
        expected_keys = set(self.nlp(expected).noun_chunks)
        actual_keys = set(self.nlp(actual).noun_chunks)
        if len(expected_keys) == 0:
            return 1.0
        return len(actual_keys.intersection(expected_keys)) / len(expected_keys)

    def _measure_consistency(self, text: str) -> float:
        """Measure internal consistency of response"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if len(sentences) <= 1:
            return 1.0
        similarities = []
        for i in range(len(sentences)-1):
            similarities.append(sentences[i].similarity(sentences[i+1]))
        return np.mean(similarities)

    def _measure_conciseness(self, text: str) -> float:
        """Measure text conciseness"""
        words = len(text.split())
        return min(1.0, 2.0 / (1 + np.exp(words / 100)))

def create_radar_chart(metrics: dict):
    """Create a radar chart of metrics"""
    # Remove overall score from radar chart
    display_metrics = {k: v for k, v in metrics.items() if k != 'overall'}
    
    fig = go.Figure(data=go.Scatterpolar(
        r=list(display_metrics.values()),
        theta=list(display_metrics.keys()),
        fill='toself'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )
    return fig

def main():
    st.set_page_config(page_title="Prompt Quality Tester", layout="wide")
    st.title("Prompt Quality Tester")

    # Sidebar for API key and instructions
    with st.sidebar:
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        st.markdown("### Task Types")
        st.write("""
        - Explain a Technical Concept
        - Generate Code Documentation
        - Write a Product Description
        - Create a Tutorial
        """)
        
        st.markdown("### Metrics Explanation")
        st.write("""
        - **Clarity**: Readability and comprehension
        - **Relevance**: Match with expected pattern
        - **Completeness**: Coverage of required elements
        - **Consistency**: Internal coherence
        - **Conciseness**: Information density
        """)

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Initialize tester
    tester = PromptQualityTester(api_key)

    # Main interface
    col1, col2 = st.columns(2)

    with col1:
        task_type = st.selectbox("Select Task Type", [
            "Explain a Technical Concept",
            "Generate Code Documentation",
            "Write a Product Description",
            "Create a Tutorial"
        ])

        example_patterns = {
            "Explain a Technical Concept": "Clear explanation with definition, examples, and use cases",
            "Generate Code Documentation": "Function purpose, parameters, return values, and usage example",
            "Write a Product Description": "Features, benefits, specifications, and target audience",
            "Create a Tutorial": "Step-by-step instructions with prerequisites and expected outcomes"
        }

        prompt = st.text_area("Enter your prompt:", height=150)
        expected = st.text_area(
            "Enter expected response pattern:",
            value=example_patterns[task_type],
            height=100
        )

    if st.button("Test Prompt"):
        if prompt and expected:
            with st.spinner("Testing prompt..."):
                result = tester.test_prompt(prompt, expected)
                
                if result:
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Quality Metrics")
                        st.metric("Overall Score", f"{result['metrics']['overall']:.2f}")
                        
                    with col2:
                        st.plotly_chart(create_radar_chart(result['metrics']))
                    
                    # Display response
                    st.subheader("Generated Response")
                    st.write(result['response'])
                    
                    # Detailed metrics
                    st.subheader("Detailed Metrics")
                    metrics_cols = st.columns(5)
                    for i, (metric, value) in enumerate(result['metrics'].items()):
                        if metric != 'overall':
                            metrics_cols[i].metric(metric.capitalize(), f"{value:.2f}")

if __name__ == "__main__":
    main()

