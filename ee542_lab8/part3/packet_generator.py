import argparse
import math
import random
import struct
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, UDP, Raw, wrpcap


WIDTH = 224
HEIGHT = 224
CHANNELS = 3
FRAME_BYTES = WIDTH * HEIGHT * CHANNELS
CHUNK_SIZE = 1400
NUMBER_OF_IMAGES = 16

MAGIC = 0x4C414238
SOURCE_MAC = "02:00:00:00:00:01"
DESTINATION_MAC = "02:00:00:00:00:02"
SOURCE_IP = "10.0.0.1"
DESTINATION_IP = "10.0.0.2"
SOURCE_PORT = 40000
DESTINATION_PORT = 5000

HEADER_FORMAT = "!IIHHIHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def create_image(frame_id):
    y, x = np.indices((HEIGHT, WIDTH))

    red = (x + frame_id * 17) % 256
    green = (y + frame_id * 31) % 256
    blue = ((x + y) // 2 + frame_id * 47) % 256

    image = np.stack((red, green, blue), axis=2)
    return image.astype(np.uint8)


def create_packets(shuffle_packets):
    packets = []
    originals_directory = Path("original_images")
    originals_directory.mkdir(exist_ok=True)

    all_original_bytes = bytearray()
    segments_per_image = math.ceil(FRAME_BYTES / CHUNK_SIZE)

    for frame_id in range(NUMBER_OF_IMAGES):
        image = create_image(frame_id)
        image_bytes = image.tobytes(order="C")

        if len(image_bytes) != FRAME_BYTES:
            raise RuntimeError("Incorrect image size")

        original_file = originals_directory / f"frame_{frame_id:02d}.bin"
        original_file.write_bytes(image_bytes)
        all_original_bytes.extend(image_bytes)

        for sequence_number in range(segments_per_image):
            offset = sequence_number * CHUNK_SIZE
            chunk = image_bytes[offset:offset + CHUNK_SIZE]
            chunk_length = len(chunk)

            header = struct.pack(
                HEADER_FORMAT,
                MAGIC,
                frame_id,
                sequence_number,
                segments_per_image,
                offset,
                chunk_length,
                FRAME_BYTES,
            )

            payload = header + chunk

            ip_packet_size = 20 + 8 + len(payload)
            if ip_packet_size > 1500:
                raise RuntimeError(
                    f"Packet exceeds MTU: {ip_packet_size} bytes"
                )

            packet = (
                Ether(src=SOURCE_MAC, dst=DESTINATION_MAC)
                / IP(src=SOURCE_IP, dst=DESTINATION_IP)
                / UDP(sport=SOURCE_PORT, dport=DESTINATION_PORT)
                / Raw(payload)
            )

            packets.append(packet)

    Path("original_images.bin").write_bytes(all_original_bytes)

    if shuffle_packets:
        random.Random(42).shuffle(packets)

    return packets, segments_per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="images.pcap",
        help="Output PCAP filename",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the packet order",
    )
    args = parser.parse_args()

    packets, segments_per_image = create_packets(args.shuffle)
    wrpcap(args.output, packets)

    print(f"Images: {NUMBER_OF_IMAGES}")
    print(f"Bytes per image: {FRAME_BYTES}")
    print(f"Image bytes per packet: {CHUNK_SIZE}")
    print(f"Custom header size: {HEADER_SIZE}")
    print(f"Packets per image: {segments_per_image}")
    print(f"Total packets: {len(packets)}")
    print(f"Shuffled: {args.shuffle}")
    print(f"Created: {args.output}")
    print("Created: original_images.bin")
    print("Created: original_images/frame_00.bin through frame_15.bin")


if __name__ == "__main__":
    main()