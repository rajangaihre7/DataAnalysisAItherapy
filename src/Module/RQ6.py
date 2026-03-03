import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def display(df, scale_map=None):
    """Display RQ6: Response Time Trends"""
    st.subheader("RQ6: Response Time Improvement")
    response_col = "What is the response time of the participant? (in minutes roughly)"
    st.markdown("**Q15:** " + response_col)
    
    # Check if response_seconds column exists, if not parse it
    if 'response_seconds' not in df.columns:
        import numpy as np
        import re
        
        def parse_response_time(val):
            if pd.isna(val):
                return np.nan
            val = str(val).lower().strip()
            if any(x in val for x in ['null', 'nothing', 'not response']):
                return np.nan
            # Range handling
            range_match = re.findall(r'(\d+)\s*[-to]+\s*(\d+)\s*(min|minute|minutes|sec|second|seconds)?', val)
            if range_match:
                start, end, unit = range_match[0]
                avg = (float(start) + float(end)) / 2
                if unit and 'min' in unit:
                    return avg * 60
                return avg
            # Minutes
            min_match = re.search(r'(\d+(?:\.\d+)?)\s*(min|minute|minutes)', val)
            if min_match:
                return float(min_match.group(1)) * 60
            # Seconds
            sec_match = re.search(r'(\d+(?:\.\d+)?)\s*(sec|second|seconds)', val)
            if sec_match:
                return float(sec_match.group(1))
            return np.nan
        
        if response_col in df.columns:
            df['response_seconds'] = df[response_col].apply(parse_response_time)
    
    # Calculate average response time per session
    if 'response_seconds' in df.columns:
        trend = df[df['response_seconds'].notna()].groupby('Session number')['response_seconds'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        
        # Line plot with markers
        ax.plot(trend['Session number'], trend['response_seconds'],
                marker='o', linewidth=2.5, markersize=9,
                color='#FF6B35', markerfacecolor='#FF6B35', markeredgewidth=2, markeredgecolor='white')
        
        # Annotate values above line
        for _, row in trend.iterrows():
            seconds = row['response_seconds']
            ax.text(row['Session number'], seconds + 5, f'{seconds:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FF6B35')
        
        # Styling
        ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(trend['response_seconds']) + 20)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Longitudinal Trends in Response Time Across Sessions',
                     fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add spacing
        st.write("")
        st.write("")
        
        # Breakdown by Autism Level
        st.markdown("---")
        st.markdown("#### Analysis by Autism Level")
        st.write("")
        
        if 'Autism Level' in df.columns:
            autism_trend = df[df['response_seconds'].notna()].groupby(['Session number', 'Autism Level'])['response_seconds'].mean().reset_index()
            
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            fig2.patch.set_facecolor('white')
            
            # Get unique autism levels
            autism_levels = sorted(autism_trend['Autism Level'].dropna().unique())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            color_map = {level: colors[i % len(colors)] for i, level in enumerate(autism_levels)}
            
            # Plot for each level
            for level in autism_levels:
                level_data = autism_trend[autism_trend['Autism Level'] == level]
                ax2.plot(level_data['Session number'], level_data['response_seconds'],
                        marker='o', linewidth=2, markersize=8,
                        label=f'Autism Level: {level}',
                        color=color_map[level])
                
                # Add annotations - red below, green above
                for _, row in level_data.iterrows():
                    seconds = row['response_seconds']
                    color = color_map[level]
                    # Red annotations below, green annotations above
                    if color == '#FF6B6B':  # Red
                        offset = -5
                        va = 'top'
                    elif color in ['#4ECDC4', '#98D8C8']:  # Green/Teal
                        offset = 5
                        va = 'bottom'
                    else:  # Other colors below
                        offset = -5
                        va = 'top'
                    
                    ax2.text(row['Session number'], seconds + offset, f'{seconds:.2f}',
                            ha='center', va=va, fontsize=8, color=color, fontweight='bold')
            
            ax2.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
            ax2.set_axisbelow(True)
            ax2.set_ylim(0, max(autism_trend['response_seconds']) + 20)
            
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_linewidth(1.5)
            ax2.spines['bottom'].set_linewidth(1.5)
            
            ax2.set_xlabel('Session Number', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
            ax2.set_title('Response Time Trends by Autism Level',
                         fontsize=13, fontweight='bold', pad=20)
            ax2.legend(fontsize=10, loc='upper right', frameon=False)
            
            plt.tight_layout()
            st.pyplot(fig2)
            
            # Add spacing
            st.write("")
            st.write("")
            
            # Third chart: Bar chart of average response time by Autism Level
            st.markdown("---")
            st.markdown("#### Average Response Time by Autism Level")
            st.write("")
            
            avg_by_level = df[df['response_seconds'].notna()].groupby('Autism Level')['response_seconds'].mean().reset_index()
            avg_by_level = avg_by_level.sort_values('Autism Level')
            
            fig3, ax3 = plt.subplots(figsize=(12, 7))
            fig3.patch.set_facecolor('white')
            
            bars = ax3.bar(avg_by_level['Autism Level'].astype(str), avg_by_level['response_seconds'],
                          color='#4ECDC4', edgecolor='black', linewidth=1.5, width=0.6)
            
            # Add value annotations on bars
            for idx, (level, value) in enumerate(zip(avg_by_level['Autism Level'], avg_by_level['response_seconds'])):
                ax3.text(idx, value + 3, f'{value:.2f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        color='#4ECDC4', family='sans-serif')
            
            # Styling
            ax3.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
            ax3.set_axisbelow(True)
            ax3.set_ylim(0, max(avg_by_level['response_seconds']) + 15)
            
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_linewidth(1.5)
            ax3.spines['bottom'].set_linewidth(1.5)
            
            ax3.set_xlabel('Autism Level', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
            ax3.set_title('Average Response Time by Autism Level',
                         fontsize=13, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig3)
    else:
        st.info("Response time data not available for trend analysis.")
