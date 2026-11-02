
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
NUM_IMAGES = 16
CHUNK_SIZE = 1400

MAGIC = 0x494D4738  # "IMG8"
HEADER_FORMAT = "!IIHHIHI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

OUTPUT_DIR = Path("results/part3")


def generate_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    images = []

    for frame_id in range(NUM_IMAGES):
        image = rng.integers(
            0, 256,
            size=(HEIGHT, WIDTH, CHANNELS),
            dtype=np.uint8
        )

        image_bytes = image.tobytes()
        images.append(image_bytes)

        filename = OUTPUT_DIR / f"frame_{frame_id:02d}.bin"
        filename.write_bytes(image_bytes)

    return images


def create_packets(images):
    packets = []

    for frame_id, image_bytes in enumerate(images):
        frame_bytes = len(image_bytes)
        nseg = math.ceil(frame_bytes / CHUNK_SIZE)

        for seq in range(nseg):
            offset = seq * CHUNK_SIZE
            chunk = image_bytes[offset:offset + CHUNK_SIZE]

            header = struct.pack(
                HEADER_FORMAT,
                MAGIC,
                frame_id,
                seq,
                nseg,
                offset,
                len(chunk),
                frame_bytes
            )

            payload = header + chunk

            pkt = (
                Ether(
                    src="02:00:00:00:00:01",
                    dst="02:00:00:00:00:02"
                )
                / IP(src="10.0.0.1", dst="10.0.0.2")
                / UDP(sport=40000, dport=5000)
                / Raw(payload)
            )

            packets.append(pkt)

    return packets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle packet order"
    )
    args = parser.parse_args()

    images = generate_images()
    packets = create_packets(images)

    if args.shuffle:
        random.Random(42).shuffle(packets)
        output = OUTPUT_DIR / "images_shuffled.pcap"
    else:
        output = OUTPUT_DIR / "images.pcap"

    wrpcap(str(output), packets)

    print(f"Header size: {HEADER_SIZE} bytes")
    print(f"Images: {len(images)}")
    print(f"Image size: {len(images[0])} bytes")
    print(f"Packets per image: {math.ceil(len(images[0]) / CHUNK_SIZE)}")
    print(f"Total packets: {len(packets)}")
    print(f"PCAP saved: {output}")


if __name__ == "__main__":
    main()
