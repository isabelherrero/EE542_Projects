
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

#define NUM 16
#define SIZE (224 * 224 * 3)
#define CHUNK 1400
#define NSEGS ((SIZE + CHUNK - 1) / CHUNK)
#define MAGIC 0x12345678

unsigned char images[NUM][SIZE];
int received[NUM][NSEGS];
int count[NUM];

int packets = 0;
int duplicates = 0;
int dropped = 0;


// Read values stored in network byte order
uint16_t read16(unsigned char *p) {
    return (p[0] << 8) | p[1];
}

uint32_t read32(unsigned char *p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           (p[2] << 8) | p[3];
}


// Check whether all images are complete
int complete() {
    for (int i = 0; i < NUM; i++) {
        if (count[i] != NSEGS) {
            return 0;
        }
    }
    return 1;
}


// Process one packet
void process(struct rte_mbuf *buf) {
    packets++;

    int size = rte_pktmbuf_pkt_len(buf);
    if (size < 64 || rte_pktmbuf_data_len(buf) != size) {
        dropped++;
        return;
    }

    unsigned char *pkt = rte_pktmbuf_mtod(buf, unsigned char *);

    // Ethernet must contain IPv4
    if (read16(pkt + 12) != 0x0800) {
        dropped++;
        return;
    }

    unsigned char *ip = pkt + 14;
    int ihl = (ip[0] & 15) * 4;

    // Check IPv4 header
    if ((ip[0] >> 4) != 4 || ihl < 20 ||
        size < 14 + ihl + 8 + 22) {
        dropped++;
        return;
    }

    // Check UDP and reject fragmented packets
    if (ip[9] != 17 || (read16(ip + 6) & 0x3fff)) {
        dropped++;
        return;
    }

    int iplen = read16(ip + 2);
    if (iplen < ihl + 8 + 22 || size < 14 + iplen) {
        dropped++;
        return;
    }

    unsigned char *udp = ip + ihl;

    // Only accept destination port 5000
    if (read16(udp + 2) != 5000) {
        dropped++;
        return;
    }

    int udplen = read16(udp + 4);
    if (udplen < 8 + 22 || udplen > iplen - ihl) {
        dropped++;
        return;
    }

    unsigned char *hdr = udp + 8;

    uint32_t magic = read32(hdr);
    uint32_t frame = read32(hdr + 4);
    uint16_t seq = read16(hdr + 8);
    uint16_t nseg = read16(hdr + 10);
    uint32_t offset = read32(hdr + 12);
    uint16_t len = read16(hdr + 16);
    uint32_t total = read32(hdr + 18);

    // Check custom packet header
    if (magic != MAGIC || frame >= NUM || seq >= NSEGS ||
        nseg != NSEGS || total != SIZE) {
        dropped++;
        return;
    }

    int expected = SIZE - seq * CHUNK;
    if (expected > CHUNK) {
        expected = CHUNK;
    }

    if (offset != seq * CHUNK || len != expected ||
        udplen < 8 + 22 + len) {
        dropped++;
        return;
    }

    // Ignore chunks already received
    if (received[frame][seq]) {
        duplicates++;
        return;
    }

    // Put the chunk in the correct image location
    memcpy(images[frame] + offset, hdr + 22, len);

    received[frame][seq] = 1;
    count[frame]++;
}


int main(int argc, char **argv) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL init failed\n");

    const char *name = "frames_rx.bin";
    if (argc > ret + 1)
        name = argv[ret + 1];

    struct rte_mempool *pool = rte_pktmbuf_pool_create("mbufs", 8191,
        256, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    if (pool == NULL)
        rte_exit(EXIT_FAILURE, "Mempool failed\n");

    if (rte_eth_dev_count_avail() == 0)
        rte_exit(EXIT_FAILURE, "No DPDK port found\n");

    struct rte_eth_conf conf = {0};

    if (rte_eth_dev_configure(0, 1, 1, &conf) < 0)
        rte_exit(EXIT_FAILURE, "Port config failed\n");

    if (rte_eth_rx_queue_setup(0, 0, 1024, rte_socket_id(), NULL, pool) < 0)
        rte_exit(EXIT_FAILURE, "RX setup failed\n");

    if (rte_eth_tx_queue_setup(0, 0, 1024, rte_socket_id(), NULL) < 0)
        rte_exit(EXIT_FAILURE, "TX setup failed\n");

    if (rte_eth_dev_start(0) < 0)
        rte_exit(EXIT_FAILURE, "Port start failed\n");

    struct rte_mbuf *bufs[32];

    // Read packets until complete or no more packets arrive
    time_t last = time(NULL);

    for (;;) {
        uint16_t n = rte_eth_rx_burst(0, 0, bufs, 32);

        if (n > 0)
            last = time(NULL);

        for (uint16_t i = 0; i < n; i++) {
            process(bufs[i]);
            rte_pktmbuf_free(bufs[i]);
        }

        if (complete() || time(NULL) - last >= 3)
            break;
    }

    // Count complete images and missing chunks
    int done = 0;
    int missing = 0;

    for (int i = 0; i < NUM; i++) {
        if (count[i] == NSEGS)
            done++;

        missing += NSEGS - count[i];
    }

    // Write the reconstructed images
    if (done == NUM) {
        FILE *f = fopen(name, "wb");

        if (f == NULL) {
            perror("File error");
            return 1;
        }

        for (int i = 0; i < NUM; i++)
            fwrite(images[i], 1, SIZE, f);

        fclose(f);
    }

    printf("Packets received: %d\n", packets);
    printf("Duplicates: %d\n", duplicates);
    printf("Dropped/invalid: %d\n", dropped);
    printf("Missing chunks: %d\n", missing);
    printf("Complete images: %d/%d\n", done, NUM);

    rte_eth_dev_stop(0);
    rte_eth_dev_close(0);
    rte_eal_cleanup();

    return done == NUM ? 0 : 1;
}
