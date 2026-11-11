import requests

#url = 'http://localhost:9696/predict'
url ='https://airline-passenger-satisfaction-sxek.onrender.com/predict'

passenger = {
 'gender': 'female',
 'customer_type': 'loyal_customer',
 'age': 31,
 'type_of_travel': 'business_travel',
 'class': 'eco',
 'flight_distance': 728,
 'inflight_wifi_service': 2,
 'departure_arrival_time_convenient': 5,
 'ease_of_online_booking': 5,
 'gate_location': 5,
 'food_and_drink': 2,
 'online_boarding': 2,
 'seat_comfort': 2,
 'inflight_entertainment': 2,
 'on_board_service': 4,
 'leg_room_service': 3,
 'baggage_handling': 3,
 'checkin_service': 4,
 'inflight_service': 3,
 'cleanliness': 2,
 'departure_delay_in_minutes': 2,
 'arrival_delay_in_minutes': 0.0
 }

response = requests.post(url, json=passenger).json()
print(response)