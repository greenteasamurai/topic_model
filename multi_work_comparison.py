import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import analyze_book
from collections import Counter

def analyze_multiple_works(file_paths):
    results = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        results[file_path] = analyze_book(text)
    return results

def compare_overall_mood(results):
    mood_data = []
    for work, analysis in results.items():
        avg_sentiment = sum(chapter['vader_sentiment']['compound'] for chapter in analysis['mood_analysis']) / len(analysis['mood_analysis'])
        mood_data.append({'Work': work, 'Average Sentiment': avg_sentiment})
    
    df = pd.DataFrame(mood_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Work', y='Average Sentiment', data=df)
    plt.title('Comparison of Overall Mood Across Works')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def compare_main_characters(results):
    character_data = []
    for work, analysis in results.items():
        for character, mentions in analysis['main_characters']:
            character_data.append({'Work': work, 'Character': character, 'Mentions': mentions})
    
    df = pd.DataFrame(character_data)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Work', y='Mentions', hue='Character', data=df)
    plt.title('Comparison of Main Characters Across Works')
    plt.xticks(rotation=45)
    plt.legend(title='Character', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt

def compare_theme_distribution(results):
    theme_data = []
    for work, analysis in results.items():
        themes = Counter()
        for chapter in analysis['theme_development']:
            themes.update(dict(chapter['topics']))
        total = sum(themes.values())
        for theme, count in themes.most_common(5):
            theme_data.append({'Work': work, 'Theme': f'Theme {theme}', 'Proportion': count / total})
    
    df = pd.DataFrame(theme_data)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Work', y='Proportion', hue='Theme', data=df)
    plt.title('Comparison of Theme Distribution Across Works')
    plt.xticks(rotation=45)
    plt.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt

def generate_comparison_report(results):
    report = "# Comparative Analysis of Literary Works\n\n"
    
    for work, analysis in results.items():
        report += f"## {work}\n"
        report += f"- Total chapters: {len(analysis['chapters'])}\n"
        report += f"- Main characters: {', '.join(char for char, _ in analysis['main_characters'][:5])}\n"
        report += f"- Average sentiment: {sum(chapter['vader_sentiment']['compound'] for chapter in analysis['mood_analysis']) / len(analysis['mood_analysis']):.2f}\n"
        report += f"- Dominant themes: {', '.join(f'Theme {theme}' for theme, _ in Counter(dict(chapter['topics']) for chapter in analysis['theme_development']).most_common(3))}\n\n"
    
    return report

def compare_works(file_paths):
    results = analyze_multiple_works(file_paths)
    
    mood_comparison = compare_overall_mood(results)
    mood_comparison.savefig('mood_comparison.png', dpi=300, bbox_inches='tight')
    
    character_comparison = compare_main_characters(results)
    character_comparison.savefig('character_comparison.png', dpi=300, bbox_inches='tight')
    
    theme_comparison = compare_theme_distribution(results)
    theme_comparison.savefig('theme_comparison.png', dpi=300, bbox_inches='tight')
    
    report = generate_comparison_report(results)
    with open('comparison_report.md', 'w') as f:
        f.write(report)
    
    return results, report