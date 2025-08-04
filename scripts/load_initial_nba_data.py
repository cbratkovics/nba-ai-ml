#!/usr/bin/env python3
"""
Script to load initial NBA data into the database
"""
import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.data.pipeline import NBADataPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Load initial NBA data"""
    print("=== NBA Data Loading Script ===")
    print("This will load NBA player and game data into your database.")
    print("Note: This process may take several minutes due to API rate limits.\n")
    
    # Ask for confirmation
    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Data loading cancelled.")
        return
    
    pipeline = NBADataPipeline()
    
    try:
        # Load teams first
        print("\n1. Loading NBA teams...")
        teams_count = await pipeline.load_teams()
        print(f"   ✓ Loaded {teams_count} teams")
        
        # Load players
        print("\n2. Loading active NBA players...")
        players_count = await pipeline.load_players(only_active=True)
        print(f"   ✓ Loaded {players_count} players")
        
        # Ask about historical data
        print("\n3. Historical game data")
        print("   Loading full historical data can take 30-60 minutes.")
        load_historical = input("   Load historical data? (yes/no): ")
        
        if load_historical.lower() == 'yes':
            # Ask for sample size
            sample = input("   Load sample only? Enter number of players (or 'all'): ")
            sample_size = None if sample == 'all' else int(sample) if sample.isdigit() else 10
            
            print(f"\n   Loading game logs for {'all' if sample_size is None else sample_size} players...")
            print("   This may take a while. Please be patient...")
            
            await pipeline.load_historical_data(
                seasons=['2023-24', '2024-25'],  # Recent seasons only for faster load
                sample_players=sample_size
            )
            print("   ✓ Historical data loaded successfully")
        
        print("\n✅ Initial data loading complete!")
        print("\nNext steps:")
        print("1. Run the model training script: python api/ml/train_models.py")
        print("2. Start the API server: uvicorn api.main:app --reload")
        print("3. Test predictions at: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n❌ Error during data loading: {e}")
        logging.error(f"Data loading failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())