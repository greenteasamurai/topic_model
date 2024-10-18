import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import analyze_book
from visualization import visualize_mood_flow, visualize_emotion_distribution, visualize_character_network
from multi_work_comparison import compare_works

def main():
    st.title("Literary Work Analysis")

    analysis_mode = st.radio("Choose analysis mode:", ("Single Work", "Multiple Works"))

    if analysis_mode == "Single Work":
        uploaded_file = st.file_uploader("Choose a text file", type="txt")
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            
            if st.button("Analyze"):
                results = analyze_book(text)
                display_single_work_analysis(results)

    else:  # Multiple Works
        uploaded_files = st.file_uploader("Choose text files", type="txt", accept_multiple_files=True)
        if uploaded_files and st.button("Compare Works"):
            file_paths = [f.name for f in uploaded_files]
            results, report = compare_works(file_paths)
            display_multi_work_comparison(results, report)

def display_single_work_analysis(results):
    st.header("Analysis Results")
    
    # Display overall statistics
    st.subheader("Overall Statistics")
    st.write(f"Total chapters: {len(results['chapters'])}")
    st.write(f"Main characters: {', '.join(char for char, _ in results['main_characters'])}")
    
    # Display mood flow
    st.subheader("Mood Flow")
    fig, ax = plt.subplots(figsize=(10, 6))
    visualize_mood_flow(results['mood_analysis'], ax)
    st.pyplot(fig)
    
    # Display emotion distribution
    st.subheader("Emotion Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    visualize_emotion_distribution(results['mood_analysis'], ax)
    st.pyplot(fig)
    
    # Display character network
    st.subheader("Character Relationship Network")
    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_character_network(results['character_graph'], ax)
    st.pyplot(fig)
    
    # Display theme development
    st.subheader("Theme Development")
    theme_df = pd.DataFrame(results['theme_development'])
    st.line_chart(theme_df.set_index('chapter'))
    
    # Display key points
    st.subheader("Key Points in the Narrative")
    for chapter, description in results['key_points']:
        st.write(f"Chapter {chapter}: {description}")

def display_multi_work_comparison(results, report):
    st.header("Comparative Analysis Results")
    
    st.subheader("Overall Mood Comparison")
    st.image('mood_comparison.png')
    
    st.subheader("Main Characters Comparison")
    st.image('character_comparison.png')
    
    st.subheader("Theme Distribution Comparison")
    st.image('theme_comparison.png')
    
    st.subheader("Comparison Report")
    st.markdown(report)

if __name__ == "__main__":
    main()