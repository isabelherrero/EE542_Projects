
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>

#define NUM_FRAMES 16
#define FRAME_BYTES (224 * 224 * 3)
#define CHUNK_SIZE 1400
#define NUM_SEGMENTS ((FRAME_BYTES + CHUNK_SIZE - 1) / CHUNK_SIZE)
#define HEADER_SIZE 22
#define MAGIC 0x494D4738U
#define UDP_PORT 5000
#define MAX_PACKET 2048

static uint8_t frames[NUM_FRAMES][FRAME_BYTES];
static uint8_t received[NUM_FRAMES][NUM_SEGMENTS];
static unsigned received_count[NUM_FRAMES];

static unsigned packets_received = 0;
static unsigned packets_dropped = 0;
static unsigned duplicates = 0;
static unsigned complete_frames = 0;

static uint16_t read16(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

static uint32_t read32(const uint8_t *p) {
    return ((uint32_t)read16(p) << 16) | read16(p + 2);
}

static void process_packet(const uint8_t *pkt, size_t size) {
    packets_received++;

    // Ethernet header
    if (size < 14 || read16(pkt + 12) != 0x0800)
        goto drop;

    // IPv4 header
    const uint8_t *ip = pkt + 14;
    size_t remaining = size - 14;

    if (remaining < 20 || (ip[0] >> 4) != 4)
        goto drop;

    size_t ihl = (ip[0] & 0x0F) * 4;

    if (ihl < 20 || remaining < ihl + 8)
        goto drop;

    if (ip[9] != 17)
        goto drop;  // Not UDP

    if ((read16(ip + 6) & 0x3FFF) != 0)
        goto drop;  // Fragmented packet

    uint16_t ip_length = read16(ip + 2);

    if (ip_length < ihl + 8 || ip_length > remaining)
        goto drop;

    // UDP header
    const uint8_t *udp = ip + ihl;
    uint16_t port = read16(udp + 2);
    uint16_t udp_length = read16(udp + 4);

    if (port != UDP_PORT ||
        udp_length < 8 + HEADER_SIZE ||
        udp_length > ip_length - ihl)
        goto drop;

    // Custom image header
    const uint8_t *data = udp + 8;
    size_t payload_size = udp_length - 8;

    uint32_t magic = read32(data);
    uint32_t frame_id = read32(data + 4);
    uint16_t seq = read16(data + 8);
    uint16_t nseg = read16(data + 10);
    uint32_t offset = read32(data + 12);
    uint16_t length = read16(data + 16);
    uint32_t frame_bytes = read32(data + 18);

    if (magic != MAGIC ||
        frame_id >= NUM_FRAMES ||
        seq >= NUM_SEGMENTS ||
        nseg != NUM_SEGMENTS ||
        frame_bytes != FRAME_BYTES ||
        offset != (uint32_t)seq * CHUNK_SIZE ||
        length == 0 ||
        length > CHUNK_SIZE ||
        offset + length > FRAME_BYTES ||
        payload_size != HEADER_SIZE + length ||
        length != ((FRAME_BYTES - offset < CHUNK_SIZE)
                  ? FRAME_BYTES - offset : CHUNK_SIZE))
        goto drop;

    // Ignore duplicate chunks
    if (received[frame_id][seq]) {
        duplicates++;
        return;
    }

    memcpy(frames[frame_id] + offset,
           data + HEADER_SIZE, length);

    received[frame_id][seq] = 1;
    received_count[frame_id]++;

    if (received_count[frame_id] == NUM_SEGMENTS) {
        complete_frames++;
        printf("Frame %u complete\n", frame_id);
    }

    return;

drop:
    packets_dropped++;
}

int main(int argc, char **argv) {
    int ret = rte_eal_init(argc, argv);

    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL initialization failed\n");

    const char *output_dir =
        (ret + 1 < argc) ? argv[ret + 1] : "frames_rx";

    if (mkdir(output_dir, 0755) != 0) {
        // Existing directory is fine
        FILE *test = NULL;
        (void)test;
    }

    if (rte_eth_dev_count_avail() == 0)
        rte_exit(EXIT_FAILURE, "No DPDK port found\n");

    struct rte_mempool *pool = rte_pktmbuf_pool_create(
        "mbufs", 8191, 256, 0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );

    if (!pool)
        rte_exit(EXIT_FAILURE, "Mempool creation failed\n");

    struct rte_eth_conf conf = {0};

    if (rte_eth_dev_configure(0, 1, 1, &conf) < 0)
        rte_exit(EXIT_FAILURE, "Port configuration failed\n");

    if (rte_eth_rx_queue_setup(
            0, 0, 1024, rte_socket_id(), NULL, pool) < 0)
        rte_exit(EXIT_FAILURE, "RX setup failed\n");

    if (rte_eth_tx_queue_setup(
            0, 0, 1024, rte_socket_id(), NULL) < 0)
        rte_exit(EXIT_FAILURE, "TX setup failed\n");

    if (rte_eth_dev_start(0) < 0)
        rte_exit(EXIT_FAILURE, "Port start failed\n");

    printf("DPDK receiver started\n");

    uint64_t last_packet = rte_get_timer_cycles();
    uint64_t frequency = rte_get_timer_hz();

    for (;;) {
        struct rte_mbuf *bufs[32];
        uint16_t n = rte_eth_rx_burst(0, 0, bufs, 32);

        if (n > 0)
            last_packet = rte_get_timer_cycles();

        for (uint16_t i = 0; i < n; i++) {
            struct rte_mbuf *m = bufs[i];
            uint32_t len = rte_pktmbuf_pkt_len(m);

            uint8_t scratch[MAX_PACKET];

            if (len <= MAX_PACKET) {
                const uint8_t *pkt =
                    rte_pktmbuf_read(m, 0, len, scratch);

                if (pkt)
                    process_packet(pkt, len);
                else
                    packets_dropped++;
            } else {
                packets_dropped++;
            }

            rte_pktmbuf_free(m);
        }

        if (complete_frames == NUM_FRAMES)
            break;

        // PCAP virtual ports do not signal EOF.
        // Stop after 3 seconds without packets.
        if (rte_get_timer_cycles() - last_packet >
            frequency * 3)
            break;

        if (n == 0)
            rte_delay_us_sleep(1000);
    }

    printf("\n=== Receiver Statistics ===\n");
    printf("Packets received: %u\n", packets_received);
    printf("Packets dropped: %u\n", packets_dropped);
    printf("Duplicate chunks: %u\n", duplicates);
    printf("Complete images: %u/%d\n",
           complete_frames, NUM_FRAMES);

    for (int i = 0; i < NUM_FRAMES; i++) {
        if (received_count[i] != NUM_SEGMENTS)
            continue;

        char filename[512];
        snprintf(filename, sizeof(filename),
                 "%s/frame_%02d.bin", output_dir, i);

        FILE *f = fopen(filename, "wb");

        if (!f) {
            perror(filename);
            return 1;
        }

        fwrite(frames[i], 1, FRAME_BYTES, f);
        fclose(f);
    }

    rte_eth_dev_stop(0);
    rte_eth_dev_close(0);
    rte_eal_cleanup();

    return complete_frames == NUM_FRAMES ? 0 : 1;
}
