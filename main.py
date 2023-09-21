import requests

def fetch_upcoming_spacex_launches():
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    
    if response.status_code == 200:
        launches_data = response.json()
        return launches_data
    else:
        return None

def display_upcoming_launches(launches_data):
    print("Upcoming SpaceX Launches:")
    for idx, launch in enumerate(launches_data):
        print(f"{idx+1}. Mission Name: {launch['name']}")
        print(f"   Launch Date: {launch['date_utc']}")
        rocket = launch['rocket']
        rocket_name = rocket['name'] if 'name' in rocket else 'N/A'
        print(f"   Rocket Name: {rocket_name}")
        print("")

if __name__ == "__main__":
    launches_data = fetch_upcoming_spacex_launches()
    if launches_data:
        display_upcoming_launches(launches_data)
    else:
        print("Failed to fetch upcoming SpaceX launches.")
