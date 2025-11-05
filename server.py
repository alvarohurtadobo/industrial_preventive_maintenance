import json
import os
import random
import time
from dotenv import load_dotenv
from paho.mqtt import client as mqtt_client

# Load environment variables from .env
load_dotenv()

PY_CLIENT_ID = f'iot_srv_python_{random.randint(0, 1_000_000)}'

def connect_mqtt():
    client_id = PY_CLIENT_ID
    broker = os.getenv('MQTT_BROKER', 'broker.emqx.io')
    port = int(os.getenv('MQTT_PORT', '1883'))
    topic = os.getenv('MQTT_TOPIC', 'flutter/sensors/2')

    username = os.getenv('MQTT_USERNAME', '')
    password = os.getenv('MQTT_PASSWORD', '')

    # For reconnect
    FIRST_RECONNECT_DELAY = int(os.getenv('MQTT_FIRST_RECONNECT_DELAY', '1'))
    RECONNECT_RATE = int(os.getenv('MQTT_RECONNECT_RATE', '2'))
    MAX_RECONNECT_COUNT = int(os.getenv('MQTT_MAX_RECONNECT_COUNT', '12'))
    MAX_RECONNECT_DELAY = int(os.getenv('MQTT_MAX_RECONNECT_DELAY', '60'))

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_disconnect(client, userdata, rc, properties=None):
        print("Disconnected with result code: %s", rc)
        reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
        while reconnect_count < MAX_RECONNECT_COUNT:
            print("Reconnecting in %d seconds...", reconnect_delay)
            time.sleep(reconnect_delay)

            try:
                client.reconnect()
                print("Reconnected successfully!")
                return
            except Exception as err:
                print("%s. Reconnect failed. Retrying...", err)

            reconnect_delay *= RECONNECT_RATE
            reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
            reconnect_count += 1

        print("Reconnect failed after %s attempts. Exiting...", reconnect_count)

    # Callback for receiving messages
    def on_message(client, userdata, msg):
        # Decoding data
        data = json.loads(msg.payload.decode())
        transformed = {
            "timestamp": data["timestamp"],
            "data": data,
            "raw": data
        }
        # displaying
        print(f"Received message: {transformed}")

    # Setup MQTT
    # client_id
    client = mqtt_client.Client(client_id= client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    # client.tls_set(ca_certs='./broker.emqx.io-ca.crt')
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    # Configure authentication if credentials are provided
    if username and password:
        client.username_pw_set(username, password)

    client.connect(broker, port)
    client.subscribe(topic)
    client.on_message = on_message
    client.loop_forever()
    return client

connect_mqtt()
