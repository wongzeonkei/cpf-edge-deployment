# Jetson Nano B01 Environment

## Device

- Device: Jetson Nano B01
- L4T: R32.7.6
- Kernel: 4.9.337-tegra
- Architecture: aarch64
- CUDA: 10.2
- TensorRT: 8.2.1
- Python: 3.6.9
- trtexec: /usr/src/tensorrt/bin/trtexec
- tegrastats: /usr/bin/tegrastats
- Power mode: MAXN
- jetson_clocks: enabled

## Storage

- System disk: `/dev/mmcblk0p1`, 14G total, 1.2G available, 92% used
- External disk: `/dev/sda1`, 29G total, 21G available, mounted at `/home/nano/sdcard`
- Project directory: `/home/nano/sdcard/jetson_projects/cpf-edge-deployment`

## Notes

- TensorRT engine generated on RTX 3090 cannot be reused on Jetson Nano B01.
- TensorRT engine must be rebuilt on the target Jetson device.
- Jetson Nano B01 has limited RAM and GPU memory, so 512x512 CPF deployment may fail or be very slow.
