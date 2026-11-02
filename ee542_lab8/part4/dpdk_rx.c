#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_udp.h>

#define MAGIC 0x4C414238
#define DESTINATION_PORT 5000

#define NUMBER_OF_IMAGES 16
#define FRAME_BYTES 150528
#define CHUNK_SIZE 1400
#define SEGMENTS_PER_IMAGE 108
#define MAX_SEGMENTS 256

#define IDLE_TIMEOUT_MS 2000

struct __attribute__((packed)) image_header {
    uint32_t magic;
    uint32_t frame_id;
    uint16_t sequence_number;
    uint16_t number_of_segments;
    uint32_t offset;
    uint16_t length;
    uint32_t frame_bytes;
};

struct frame_state {
    uint8_t *data;
    bool received_segments[MAX_SEGMENTS];
    uint16_t expected_segments;
    uint16_t received_count;
    uint32_t received_bytes;
    bool complete;
};

enum packet_result {
    PACKET_DROPPED = -1,
    PACKET_IGNORED = 0,
    PACKET_ACCEPTED = 1,
    PACKET_DUPLICATE = 2
};

static enum packet_result process_packet(
    struct rte_mbuf *mbuf,
    struct frame_state frames[NUMBER_OF_IMAGES],
    uint32_t *complete_frames
) {
    uint32_t packet_length = rte_pktmbuf_pkt_len(mbuf);
    uint8_t packet_copy[2048];

    if (packet_length > sizeof(packet_copy)) {
        return PACKET_DROPPED;
    }

    const uint8_t *packet = rte_pktmbuf_read(
        mbuf,
        0,
        packet_length,
        packet_copy
    );

    if (packet == NULL) {
        return PACKET_DROPPED;
    }

    if (packet_length < sizeof(struct rte_ether_hdr)) {
        return PACKET_DROPPED;
    }

    const struct rte_ether_hdr *ethernet =
        (const struct rte_ether_hdr *)packet;

    if (rte_be_to_cpu_16(ethernet->ether_type) != RTE_ETHER_TYPE_IPV4) {
        return PACKET_IGNORED;
    }

    uint32_t ip_offset = sizeof(struct rte_ether_hdr);

    if (packet_length < ip_offset + sizeof(struct rte_ipv4_hdr)) {
        return PACKET_DROPPED;
    }

    const struct rte_ipv4_hdr *ipv4 =
        (const struct rte_ipv4_hdr *)(packet + ip_offset);

    if ((ipv4->version_ihl >> 4) != 4) {
        return PACKET_DROPPED;
    }

    uint32_t ipv4_header_length = (ipv4->version_ihl & 0x0F) * 4;

    if (ipv4_header_length < sizeof(struct rte_ipv4_hdr)) {
        return PACKET_DROPPED;
    }

    if (ipv4->next_proto_id != IPPROTO_UDP) {
        return PACKET_IGNORED;
    }

    uint32_t udp_offset = ip_offset + ipv4_header_length;

    if (packet_length < udp_offset + sizeof(struct rte_udp_hdr)) {
        return PACKET_DROPPED;
    }

    const struct rte_udp_hdr *udp =
        (const struct rte_udp_hdr *)(packet + udp_offset);

    if (rte_be_to_cpu_16(udp->dst_port) != DESTINATION_PORT) {
        return PACKET_IGNORED;
    }

    uint16_t udp_length = rte_be_to_cpu_16(udp->dgram_len);

    if (udp_length < sizeof(struct rte_udp_hdr) +
                         sizeof(struct image_header)) {
        return PACKET_DROPPED;
    }

    uint32_t payload_offset = udp_offset + sizeof(struct rte_udp_hdr);
    uint32_t payload_length = udp_length - sizeof(struct rte_udp_hdr);

    if (packet_length < payload_offset + payload_length) {
        return PACKET_DROPPED;
    }

    struct image_header header;
    memcpy(&header, packet + payload_offset, sizeof(header));

    uint32_t magic = rte_be_to_cpu_32(header.magic);
    uint32_t frame_id = rte_be_to_cpu_32(header.frame_id);
    uint16_t sequence_number =
        rte_be_to_cpu_16(header.sequence_number);
    uint16_t number_of_segments =
        rte_be_to_cpu_16(header.number_of_segments);
    uint32_t offset = rte_be_to_cpu_32(header.offset);
    uint16_t chunk_length = rte_be_to_cpu_16(header.length);
    uint32_t frame_bytes = rte_be_to_cpu_32(header.frame_bytes);

    if (magic != MAGIC) {
        return PACKET_DROPPED;
    }

    if (frame_id >= NUMBER_OF_IMAGES) {
        return PACKET_DROPPED;
    }

    if (number_of_segments != SEGMENTS_PER_IMAGE ||
        number_of_segments > MAX_SEGMENTS) {
        return PACKET_DROPPED;
    }

    if (sequence_number >= number_of_segments) {
        return PACKET_DROPPED;
    }

    if (frame_bytes != FRAME_BYTES) {
        return PACKET_DROPPED;
    }

    if (chunk_length > CHUNK_SIZE) {
        return PACKET_DROPPED;
    }

    if (offset != (uint32_t)sequence_number * CHUNK_SIZE) {
        return PACKET_DROPPED;
    }

    uint32_t expected_length = CHUNK_SIZE;

    if (sequence_number == number_of_segments - 1) {
        expected_length = FRAME_BYTES - offset;
    }

    if (chunk_length != expected_length) {
        return PACKET_DROPPED;
    }

    if (offset + chunk_length > FRAME_BYTES) {
        return PACKET_DROPPED;
    }

    if (payload_length < sizeof(struct image_header) + chunk_length) {
        return PACKET_DROPPED;
    }

    struct frame_state *frame = &frames[frame_id];

    if (frame->received_segments[sequence_number]) {
        return PACKET_DUPLICATE;
    }

    if (frame->expected_segments == 0) {
        frame->expected_segments = number_of_segments;
    } else if (frame->expected_segments != number_of_segments) {
        return PACKET_DROPPED;
    }

    const uint8_t *image_data =
        packet + payload_offset + sizeof(struct image_header);

    memcpy(frame->data + offset, image_data, chunk_length);

    frame->received_segments[sequence_number] = true;
    frame->received_count++;
    frame->received_bytes += chunk_length;

    if (!frame->complete &&
        frame->received_count == frame->expected_segments &&
        frame->received_bytes == FRAME_BYTES) {
        frame->complete = true;
        (*complete_frames)++;
        printf(
            "Frame %u complete: %u segments, %u bytes\n",
            frame_id,
            frame->received_count,
            frame->received_bytes
        );
    }

    return PACKET_ACCEPTED;
}

int main(int argc, char **argv) {
    int eal_arguments = rte_eal_init(argc, argv);

    if (eal_arguments < 0) {
        rte_exit(EXIT_FAILURE, "EAL initialization failed\n");
    }

    argc -= eal_arguments;
    argv += eal_arguments;

    const char *output_filename =
        argc > 1 ? argv[1] : "frames_rx.bin";

    if (rte_eth_dev_count_avail() == 0) {
        rte_exit(EXIT_FAILURE, "No DPDK ports found\n");
    }

    uint16_t port_id;
    RTE_ETH_FOREACH_DEV(port_id) {
        break;
    }

    struct rte_mempool *pool = rte_pktmbuf_pool_create(
        "mbufs",
        8191,
        256,
        0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );

    if (pool == NULL) {
        rte_exit(EXIT_FAILURE, "Could not create mbuf pool\n");
    }

    struct rte_eth_conf port_configuration;
    memset(&port_configuration, 0, sizeof(port_configuration));

    if (rte_eth_dev_configure(
            port_id,
            1,
            0,
            &port_configuration
        ) < 0) {
        rte_exit(EXIT_FAILURE, "Could not configure port\n");
    }

    if (rte_eth_rx_queue_setup(
            port_id,
            0,
            1024,
            rte_eth_dev_socket_id(port_id),
            NULL,
            pool
        ) < 0) {
        rte_exit(EXIT_FAILURE, "Could not configure RX queue\n");
    }

    if (rte_eth_dev_start(port_id) < 0) {
        rte_exit(EXIT_FAILURE, "Could not start port\n");
    }

    struct frame_state frames[NUMBER_OF_IMAGES];
    memset(frames, 0, sizeof(frames));

    for (uint32_t i = 0; i < NUMBER_OF_IMAGES; i++) {
        frames[i].data = calloc(FRAME_BYTES, 1);

        if (frames[i].data == NULL) {
            rte_exit(EXIT_FAILURE, "Could not allocate image memory\n");
        }
    }

    uint64_t packets_received = 0;
    uint64_t chunks_accepted = 0;
    uint64_t packets_ignored = 0;
    uint64_t packets_dropped = 0;
    uint64_t duplicate_chunks = 0;
    uint32_t complete_frames = 0;
    uint32_t idle_milliseconds = 0;

    printf("Receiving packets from DPDK port %u\n", port_id);

    while (complete_frames < NUMBER_OF_IMAGES) {
        struct rte_mbuf *buffers[32];

        uint16_t count = rte_eth_rx_burst(
            port_id,
            0,
            buffers,
            32
        );

        if (count == 0) {
            if (packets_received > 0) {
                idle_milliseconds++;

                if (idle_milliseconds >= IDLE_TIMEOUT_MS) {
                    printf("Stopped after PCAP idle timeout\n");
                    break;
                }
            }

            rte_delay_us_sleep(1000);
            continue;
        }

        idle_milliseconds = 0;
        packets_received += count;

        for (uint16_t i = 0; i < count; i++) {
            enum packet_result result = process_packet(
                buffers[i],
                frames,
                &complete_frames
            );

            if (result == PACKET_ACCEPTED) {
                chunks_accepted++;
            } else if (result == PACKET_DUPLICATE) {
                duplicate_chunks++;
            } else if (result == PACKET_IGNORED) {
                packets_ignored++;
            } else {
                packets_dropped++;
            }

            rte_pktmbuf_free(buffers[i]);
        }
    }

    FILE *output = fopen(output_filename, "wb");

    if (output == NULL) {
        rte_exit(EXIT_FAILURE, "Could not open output file\n");
    }

    for (uint32_t i = 0; i < NUMBER_OF_IMAGES; i++) {
        if (frames[i].complete) {
            size_t written = fwrite(
                frames[i].data,
                1,
                FRAME_BYTES,
                output
            );

            if (written != FRAME_BYTES) {
                fclose(output);
                rte_exit(EXIT_FAILURE, "Could not write output file\n");
            }
        }
    }

    fclose(output);

    printf("\nReceiver statistics\n");
    printf("Packets received: %lu\n", packets_received);
    printf("Chunks accepted: %lu\n", chunks_accepted);
    printf("Packets ignored: %lu\n", packets_ignored);
    printf("Packets dropped: %lu\n", packets_dropped);
    printf("Duplicate chunks: %lu\n", duplicate_chunks);
    printf("Complete images: %u/%u\n",
           complete_frames, NUMBER_OF_IMAGES);
    printf("Output file: %s\n", output_filename);

    for (uint32_t i = 0; i < NUMBER_OF_IMAGES; i++) {
        free(frames[i].data);
    }

    rte_eth_dev_stop(port_id);
    rte_eth_dev_close(port_id);
    rte_eal_cleanup();

    return complete_frames == NUMBER_OF_IMAGES
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}