import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
from mplsoccer import VerticalPitch, add_image, Pitch
import cmasher as cmr
import numpy as np
from scipy.ndimage import gaussian_filter

# Set page config
st.set_page_config(page_title="Miki-Analytics-Hackathon", layout="wide", initial_sidebar_state="expanded")

# Apply custom theme settings
# Apply custom theme settings
st.markdown(
    """
    <style>
        body {background-color: #1e1e1e; color: #e0e0e0;}
        .st-bd {color: #4caf50 !important;} /* Accent color for headings or borders */
        
        div[role="tablist"] button {
            border-radius: 5px; 
            color: #e0e0e0;
            background-color: #021401;
            border: 2px solid transparent; /* Default border color to remove orange */
        }

        .css-1aumxhk {background-color: #1e1e1e;}

        .stTabs [data-baseweb="tab-highlight"] {
        background-color:#4caf50;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Miki-Analytics-Hackathon")

# Load team pressure data from pressure_predictions.csv when the site is loaded
try:
    team_data = pd.read_csv('pressure_predictions.csv')
except FileNotFoundError:
    team_data = None

# Create main tabs for Team Pressure Analysis and Individual Player Analysis
tab1, tab2 = st.tabs(["Team Pressure Analysis", "Individual Player Analysis"])

# Team Pressure Analysis tab
with tab1:
    st.header("Team Pressure Analysis")
    st.write("Analyze team pressure statistics and get insights on the team's performance.")

    if team_data is not None:
        # Filters Section
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                # Dropdown to select a competition
                competition_names = team_data["competition_name"].unique()
                selected_competition = st.selectbox("Select a Competition", competition_names)

                # Filter data for the selected competition
                competition_filtered_data = team_data[team_data["competition_name"] == selected_competition]

            with col2:
                # Dropdown to select a team
                team_names = sorted(competition_filtered_data["team_name"].unique())

                selected_team = st.selectbox("Select a Team", team_names)

                # Filter data for the selected team
                team_filtered_data = competition_filtered_data[competition_filtered_data["team_name"] == selected_team]
                df_false9 = team_filtered_data[['player_name','x', 'y','Foul', 'ball_recovery', 'moved_closer_to_goal', 'moved_further_from_goal', 'shot_conceded']].copy()

        # Make filters sticky
        st.markdown("<style>div.sticky-filter {position: -webkit-sticky; position: sticky; top: 0; z-index: 1; background: #2e2e2e; padding: 10px;}</style>", unsafe_allow_html=True)

        # Team Pressure Stats Section
        st.subheader("Team Pressure Statistics")
        # Calculate the amount of pressures and matches played for each team in the selected competition
        team_pressures = competition_filtered_data.groupby("team_name")["id"].nunique().reset_index(name="pressure_count")
        team_matches = competition_filtered_data.groupby("team_name")["match_id"].nunique().reset_index(name="matches_played")
        team_stats = pd.merge(team_pressures, team_matches, on="team_name")
        team_stats["pressures_per_match"] = team_stats["pressure_count"] / team_stats["matches_played"]

        # Sort the team_stats dataframe by pressures_per_match
        team_stats = team_stats.sort_values(by="pressures_per_match", ascending=False).reset_index(drop=True)

        # Filter data for the selected team
        selected_team_stats = team_stats[team_stats["team_name"] == selected_team]
        pressure_count = selected_team_stats["pressure_count"].values[0]
        pressures_per_match = selected_team_stats["pressures_per_match"].values[0]

        # Display the team statistics as a list
        st.write("- The selected team made {} distinct pressures.".format(pressure_count))
        st.write("- The selected team, {}, averaged {:.2f} pressures per match.".format(selected_team, pressures_per_match))

        # Find the player who made the most pressures per match in the selected team
        player_pressures = team_filtered_data.groupby("player_name")["id"].nunique().reset_index(name="pressure_count")
        player_matches = team_filtered_data.groupby("player_name")["match_id"].nunique().reset_index(name="matches_played")
        player_stats = pd.merge(player_pressures, player_matches, on="player_name")
        player_stats["pressures_per_match"] = player_stats["pressure_count"] / player_stats["matches_played"]

        # Get the player with the most pressures per match
        top_player = player_stats.sort_values(by="pressures_per_match", ascending=False).iloc[0]
        st.write("- The player with the most pressures per match in {} is {} with an average of {:.2f} pressures per match.".format(selected_team, top_player['player_name'], top_player['pressures_per_match']))

        # Divider between sections
        st.markdown("---")

        # Plot Section
        st.subheader("Team Pressure Insights")
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart: Compare pressures per match for all teams, highlighting the selected team
            st.subheader(f"Pressures Per Match Comparison for {selected_team}")
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.bar(team_stats.index, team_stats["pressures_per_match"], color='#A9A9A9')

            # Set team names as x-tick labels
            ax.set_xticks(team_stats.index)
            ax.set_xticklabels(team_stats["team_name"], rotation=90, color='white')

            # Highlight the selected team
            selected_index = team_stats[team_stats["team_name"] == selected_team].index[0]
            bars[selected_index].set_color('#347244')  # Light purple color

            # Add labels to bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=8, color='white')

            # Set dark theme styles
            ax.set_facecolor('#2e2e2e')
            fig.patch.set_facecolor('#2e2e2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            plt.ylabel("Pressures per Match", color='white')
            plt.title("Team Pressures Per Match Comparison", color='white')
            st.pyplot(fig)

            with col2:
                # Team pressure map
                st.subheader(f"Team Pressure Map for {selected_team}")

                # Create the pitch
                pitch = Pitch(pitch_type='statsbomb', line_color='#e0e0e0', line_zorder=2, pitch_color='#2e2e2e')

                # Create the heatmap with increased smoothing
                fig, ax = pitch.draw(figsize=(10, 8))  # Match the same vertical size as the other plot
                heatmap, xedges, yedges = np.histogram2d(df_false9['x'], df_false9['y'], bins=[50, 50])
                heatmap = gaussian_filter(heatmap, sigma=5)  # Apply more Gaussian smoothing
                pcm = ax.imshow(heatmap.T, extent=(0, pitch.dim.pitch_length, 0, pitch.dim.pitch_width), aspect='auto', cmap='Greens', alpha=0.6, zorder=1)

                # Set dark theme styles
                fig.patch.set_facecolor('#2e2e2e')
                ax.set_facecolor('#2e2e2e')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')

                # Adjust layout to make space for the footer
                plt.subplots_adjust(bottom=0.1)

                # Add the direction of play as a footer at the bottom center of the figure
                fig.text(0.5, 0.0025, 'Direction of Play →', color='white', ha='center', fontsize=12)

                st.pyplot(fig)
                
        # Divider between sections
        st.markdown("---")

        # Model Results Section
        st.subheader("Model Results")
        st.write("The selected team has undergone analysis using a machine learning model. Here are the key visualizations.")

        # Define columns for weights
        weight_columns = ['Foul', 'ball_recovery', 'moved_closer_to_goal', 'moved_further_from_goal', 'shot_conceded']

        # Set up Streamlit container to display plots side by side
        cols = st.columns(5)

        for col, weight_col in zip(cols, weight_columns):
            with col:
                fig, ax = plt.subplots(figsize=(8, 12))  # Individual plot size
                pitch = VerticalPitch(pitch_type='statsbomb', line_color='#e0e0e0', line_zorder=2, pitch_color='#2e2e2e')
                pitch.draw(ax=ax)

                # Ensure weights are valid
                weights = df_false9[weight_col].fillna(0).clip(lower=0)
                print(weights.shape)
                print(df_false9.shape)
                # Create a weighted 2D histogram based on the current weight column
                heatmap, xedges, yedges = np.histogram2d(
                    df_false9['y'], df_false9['x'], 
                    bins=[50, 50], 
                    weights=weights
                )

                # Apply Gaussian smoothing to the weighted heatmap
                if np.max(heatmap) > 0:
                    heatmap = gaussian_filter(heatmap, sigma=5)
                else:
                    print(f"Warning: All heatmap values are zero after applying weights for {weight_col}.")

                # Plot the heatmap on the pitch
                ax.imshow(
                    heatmap.T, 
                    extent=(0, pitch.dim.pitch_width, 0, pitch.dim.pitch_length), 
                    aspect='auto', 
                    cmap='Greens', 
                    alpha=0.6, 
                    zorder=1
                )

                # Set title for each pitch to indicate the weight column
                ax.set_title(weight_col.replace('_', ' ').title(), color='white', fontsize=14)

                # Set dark theme styles
                fig.patch.set_facecolor('#2e2e2e')
                ax.set_facecolor('#2e2e2e')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')

                # Display the plot in the current column
                st.pyplot(fig)

        # Add direction of play note at the bottom
        st.write("**Note:** The direction of play is from top to the bottom of each pitch.")
        # Add Team Ranking Section
        st.subheader("Team Rankings Per Match")
        st.markdown(
            """
            <style>
            .dataframe-container {
                width: 100%;
            }
            .stDataFrame div {
                overflow: scroll;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Create a filter for competition_name
        competition_names_ranking = team_data["competition_name"].unique()
        selected_competition_ranking = st.selectbox("Select a Competition for Team Rankings", competition_names_ranking)

        # Filter data for the selected competition
        competition_filtered_data_ranking = team_data[team_data["competition_name"] == selected_competition_ranking]

        # Check if data exists for the selected competition
        if not competition_filtered_data_ranking.empty:
            # Select the column to calculate ranking from
            ranking_metric = st.selectbox("Select Metric for Team Rankings", ['Foul', 'ball_recovery', 'moved_closer_to_goal', 'moved_further_from_goal', 'shot_conceded'])

            # Group data by competition_name, team_name
            team_ranking_data = competition_filtered_data_ranking.groupby(["competition_name", "team_name"]).agg(
                total_metric=(ranking_metric, 'sum'),
                matches_played=("match_id", pd.Series.nunique)
            ).reset_index()

            # Calculate rankings per match (total_metric / matches_played)
            team_ranking_data["metric_per_match"] = team_ranking_data["total_metric"] / team_ranking_data["matches_played"]

            # Sort the data by the calculated metric
            team_ranking_data = team_ranking_data.sort_values(by="metric_per_match", ascending=False).reset_index(drop=True)

            # Rename columns for better formatting (removing underscores and capitalizing)
            team_ranking_data = team_ranking_data.rename(columns=lambda x: x.replace('_', ' ').title())

            # Display the ranking table
            st.write(f"Team Rankings based on {ranking_metric.replace('_', ' ').title()} per Match:")
            
            # Format the table to span the entire width
            st.dataframe(team_ranking_data.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center')]
            }]), use_container_width=True)
        else:
            st.write("No data available for the selected competition.")

        # Add StatsBomb logo at the bottom
        SB_LOGO_URL = ('https://raw.githubusercontent.com/statsbomb/open-data/'
                    'master/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png')
        sb_logo = Image.open(urlopen(SB_LOGO_URL))
        st.image(sb_logo, width=100, caption="StatsBomb")

# Individual Player Analysis tab
with tab2:
    st.header("Team Pressure Analysis")
    st.write("Analyze team pressure statistics and get insights on the team's performance.")

    if team_data is not None:
        # Filters Section
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                # Dropdown to select a competition
                competition_names = team_data["competition_name"].unique()
                selected_competition = st.selectbox("Select a Competition for the Player", competition_names)

                # Filter data for the selected competition
                competition_filtered_data = team_data[team_data["competition_name"] == selected_competition]

            with col2:
                # Dropdown to select a team
                team_names = sorted(competition_filtered_data["team_name"].unique())
                selected_team = st.selectbox("Select a Team for the Player", team_names)

                # Filter data for the selected team
                team_filtered_data = competition_filtered_data[competition_filtered_data["team_name"] == selected_team]
            
            # Adding player filter
            player_names = sorted(team_filtered_data["player_name"].unique())
            if len(player_names) > 0:
                selected_player = st.selectbox("Select a Player", player_names)

                # Filter data for the selected player
                player_filtered_data = team_filtered_data[team_filtered_data["player_name"] == selected_player]

                df_false9 = player_filtered_data[['player_name','x', 'y', 'Foul', 'ball_recovery', 'moved_closer_to_goal', 'moved_further_from_goal', 'shot_conceded']].copy()
            else:
                st.write("No players available for the selected filters.")

        # Make filters sticky
        st.markdown(
            "<style>div.sticky-filter {position: -webkit-sticky; position: sticky; top: 0; z-index: 1; background: #2e2e2e; padding: 10px;}</style>", 
            unsafe_allow_html=True
        )

        # Team Pressure Stats Section
        st.subheader("Player Pressure Statistics")
        # Calculate the amount of pressures and matches played for each team in the selected competition
        team_pressures = competition_filtered_data.groupby("team_name")["id"].nunique().reset_index(name="pressure_count")
        team_matches = competition_filtered_data.groupby("team_name")["match_id"].nunique().reset_index(name="matches_played")
        team_stats = pd.merge(team_pressures, team_matches, on="team_name")
        team_stats["pressures_per_match"] = team_stats["pressure_count"] / team_stats["matches_played"]

        # Sort the team_stats dataframe by pressures_per_match
        team_stats = team_stats.sort_values(by="pressures_per_match", ascending=False).reset_index(drop=True)

        # Filter data for the selected team
        selected_team_stats = team_stats[team_stats["team_name"] == selected_team]
        if not selected_team_stats.empty:
            pressure_count = selected_team_stats["pressure_count"].values[0]
            pressures_per_match = selected_team_stats["pressures_per_match"].values[0]

            # Display the team statistics as a list
            st.write("- The selected team made {} distinct pressures.".format(pressure_count))
            st.write("- The selected team, {}, averaged {:.2f} pressures per match.".format(selected_team, pressures_per_match))
        else:
            st.write("No pressure data available for the selected team.")

        # Find the player who made the most pressures per match in the selected team
        player_pressures = team_filtered_data.groupby("player_name")["id"].nunique().reset_index(name="pressure_count")
        player_matches = team_filtered_data.groupby("player_name")["match_id"].nunique().reset_index(name="matches_played")
        player_stats = pd.merge(player_pressures, player_matches, on="player_name")
        player_stats["pressures_per_match"] = player_stats["pressure_count"] / player_stats["matches_played"]

        if not player_stats.empty:
            # Get the player with the most pressures per match
            top_player = player_stats.sort_values(by="pressures_per_match", ascending=False).iloc[0]
            st.write("- The player with the most pressures per match in {} is {} with an average of {:.2f} pressures per match.".format(selected_team, top_player['player_name'], top_player['pressures_per_match']))
        else:
            st.write("No player pressure data available for the selected team.")

        # Divider between sections
        st.markdown("---")

        # Plot Section
        st.subheader("Player Pressure Insights")
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart: Compare pressures per match for all players in the competition, highlighting the selected player
            st.subheader(f"Pressures Per Match Comparison for {selected_competition}")
            competition_player_pressures = competition_filtered_data.groupby("player_name")["id"].nunique().reset_index(name="pressure_count")
            competition_player_matches = competition_filtered_data.groupby("player_name")["match_id"].nunique().reset_index(name="matches_played")
            competition_player_stats = pd.merge(competition_player_pressures, competition_player_matches, on="player_name")
            competition_player_stats["pressures_per_match"] = competition_player_stats["pressure_count"] / competition_player_stats["matches_played"]

            # Sort the competition_player_stats dataframe by pressures_per_match
            competition_player_stats = competition_player_stats.sort_values(by="pressures_per_match", ascending=False).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.bar(competition_player_stats.index, competition_player_stats["pressures_per_match"], color='#A9A9A9')

            # Set player names as x-tick labels
            ax.set_xticks(competition_player_stats.index)
            ax.set_xticklabels([])  # Do not label all columns, only the special one

            # Highlight the selected player
            if selected_player in competition_player_stats["player_name"].values:
                selected_index = competition_player_stats[competition_player_stats["player_name"] == selected_player].index[0]
                bars[selected_index].set_color('#347244')  # Highlight selected player

                # Add label only to the highlighted bar
                yval = bars[selected_index].get_height()
                ax.text(bars[selected_index].get_x() + bars[selected_index].get_width() / 2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=8, color='white')

            # Set dark theme styles
            ax.set_facecolor('#2e2e2e')
            fig.patch.set_facecolor('#2e2e2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            plt.ylabel("Pressures per Match", color='white')
            plt.title("Player Pressures Per Match Comparison", color='white')
            st.pyplot(fig)

        with col2:
            # Team pressure map
            st.subheader(f"Team Pressure Map for {selected_player}")

            # Create the pitch
            pitch = Pitch(pitch_type='statsbomb', line_color='#e0e0e0', line_zorder=2, pitch_color='#2e2e2e')

            # Create the heatmap with increased smoothing
            fig, ax = pitch.draw(figsize=(10, 8))  # Match the same vertical size as the other plot
            if not player_filtered_data.empty:
                heatmap, xedges, yedges = np.histogram2d(df_false9['x'], df_false9['y'], bins=[50, 50])
                heatmap = gaussian_filter(heatmap, sigma=5)  # Apply more Gaussian smoothing
                pcm = ax.imshow(heatmap.T, extent=(0, pitch.dim.pitch_length, 0, pitch.dim.pitch_width), aspect='auto', cmap='Greens', alpha=0.6, zorder=1)

            # Set dark theme styles
            fig.patch.set_facecolor('#2e2e2e')
            ax.set_facecolor('#2e2e2e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')

            # Adjust layout to make space for the footer
            plt.subplots_adjust(bottom=0.1)

            # Add the direction of play as a footer at the bottom center of the figure
            fig.text(0.5, 0.0025, 'Direction of Play →', color='white', ha='center', fontsize=12)

            st.pyplot(fig)
                
        # Divider between sections
        st.markdown("---")

        # Model Results Section
        st.subheader("Model Results")
        st.write("The selected team has undergone analysis using a machine learning model. Here are the key visualizations.")

        # Define columns for weights
        weight_columns = ['Foul', 'ball_recovery', 'moved_closer_to_goal', 'moved_further_from_goal', 'shot_conceded']

        # Set up Streamlit container to display plots side by side
        cols = st.columns(5)

        for col, weight_col in zip(cols, weight_columns):
            with col:
                fig, ax = plt.subplots(figsize=(8, 12))  # Individual plot size
                pitch = VerticalPitch(pitch_type='statsbomb', line_color='#e0e0e0', line_zorder=2, pitch_color='#2e2e2e')
                pitch.draw(ax=ax)

                # Ensure weights are valid
                weights = df_false9[weight_col].fillna(0).clip(lower=0)
                print(df_false9.head())
                if weights.sum() > 0:
                    # Create a weighted 2D histogram based on the current weight column
                    heatmap, xedges, yedges = np.histogram2d(
                        df_false9['y'], df_false9['x'],
                        bins=[50, 50],
                        weights=weights
                    )

                    # Apply Gaussian smoothing to the weighted heatmap
                    heatmap = gaussian_filter(heatmap, sigma=5)

                    # Plot the heatmap on the pitch
                    ax.imshow(
                        heatmap.T, 
                        extent=(0, pitch.dim.pitch_width, 0, pitch.dim.pitch_length), 
                        aspect='auto', 
                        cmap='Greens', 
                        alpha=0.6, 
                        zorder=1
                    )
                else:
                    st.write(f"No data available for {weight_col}.")

                # Set title for each pitch to indicate the weight column
                ax.set_title(weight_col.replace('_', ' ').title(), color='white', fontsize=14)

                # Set dark theme styles
                fig.patch.set_facecolor('#2e2e2e')
                ax.set_facecolor('#2e2e2e')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')

                # Display the plot in the current column
                st.pyplot(fig)

        # Add custom CSS to make the table span the full width
        st.markdown(
            """
            <style>
            .dataframe-container {
                width: 100%;
            }
            .stDataFrame div {
                overflow: scroll;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Add Ranking Section
        st.subheader("Player Rankings Per Match")

        # Create a filter for competition_name
        competition_names_ranking = team_data["competition_name"].unique()
        selected_competition_ranking = st.selectbox("Select a Competition for Rankings", competition_names_ranking)

        # Filter data for the selected competition
        competition_filtered_data_ranking = team_data[team_data["competition_name"] == selected_competition_ranking]

        # Check if data exists for the selected competition
        if not competition_filtered_data_ranking.empty:
            # Select the column to calculate ranking from
            ranking_metric = st.selectbox("Select Metric for Rankings", ['Foul', 'ball_recovery', 'moved_closer_to_goal', 'moved_further_from_goal', 'shot_conceded'])

            # Group data by competition_name, team_name, player_name
            ranking_data = competition_filtered_data_ranking.groupby(["competition_name", "team_name", "player_name"]).agg(
                total_metric=(ranking_metric, 'sum'),
                matches_played=("match_id", pd.Series.nunique)
            ).reset_index()

            # Dropdown to select the minimum number of matches played
            min_matches = st.slider("Select Minimum Number of Matches Played", min_value=int(ranking_data["matches_played"].min()), max_value=int(ranking_data["matches_played"].max()), value=1)

            # Filter the data based on the selected number of matches played
            ranking_data = ranking_data[ranking_data["matches_played"] >= min_matches]

            # Calculate rankings per match (total_metric / matches_played)
            ranking_data["metric_per_match"] = ranking_data["total_metric"] / ranking_data["matches_played"]

            # Sort the data by the calculated metric
            ranking_data = ranking_data.sort_values(by="metric_per_match", ascending=False).reset_index(drop=True)

            # Rename columns for better formatting (removing underscores and capitalizing)
            ranking_data = ranking_data.rename(columns=lambda x: x.replace('_', ' ').title())

            # Display the ranking table
            st.write(f"Player Rankings based on {ranking_metric.replace('_', ' ').title()} per Match (Minimum {min_matches} Matches):")
            
            # Format the table to span the entire width
            st.dataframe(ranking_data.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center')]
            }]), use_container_width=True)
        else:
            st.write("No data available for the selected competition.")




        # Add direction of play note at the bottom
        st.write("**Note:** The direction of play is from top to bottom of each pitch.")

        # Add StatsBomb logo at the bottom
        SB_LOGO_URL = ('https://raw.githubusercontent.com/statsbomb/open-data/'
                    'master/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png')
        try:
            sb_logo = Image.open(urlopen(SB_LOGO_URL))
            st.image(sb_logo, width=100, caption="StatsBomb")
        except:
            st.write("Unable to load StatsBomb logo.")
