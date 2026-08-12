"""Secure shell network client interface for distributed cluster hardware.

This module handles low-level SSH tunnels and interactive session transport
layers mapping directly to local host configurations. It isolates network logic
to poll individual server utilization states, target concrete GPU resources,
and control background execution streams.
"""

from pathlib import Path

import paramiko


class RemoteClusterNode:
    """Manages secure communication interfaces with a remote cluster machine."""

    def __init__(self, host_name: str) -> None:
        """Initializes client configurations matching ssh config directives.

        Args:
            host_name: Name alias defined inside user ~/.ssh/config profiles.
        """
        self.host_name = host_name
        self.config_path: Path = Path.home() / ".ssh" / "config"
        self.known_hosts: Path = Path.home() / ".ssh" / "known_hosts"

    def _establish_client(self) -> paramiko.SSHClient:
        """Assembles configured SSH client from profile configuration matches.

        Returns:
            paramiko.SSHClient: An authenticated, active connection pipe.
        """
        ssh_config = paramiko.SSHConfig()
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as config_file:
                ssh_config.parse(config_file)

        lookup_map = ssh_config.lookup(self.host_name)
        client = paramiko.SSHClient()

        if self.known_hosts.exists():
            client.load_host_keys(str(self.known_hosts))
        else:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connection_kwargs = {
            "hostname": lookup_map.get("hostname", self.host_name),
            "username": lookup_map.get("user"),
            "port": int(lookup_map.get("port", 22)),
            "timeout": 10,
        }

        if "identityfile" in lookup_map:
            # Identityfile entries are often stored as list arrays by paramiko's lookup mechanism
            id_file_entry: str = lookup_map["identityfile"]
            raw_path: str = (
                id_file_entry[0] if isinstance(id_file_entry, list) else id_file_entry
            )

            # Resolve path strings starting with standard user home shortcuts tilde (~)
            identity_path: Path = Path(raw_path).expanduser()
            connection_kwargs["key_filename"] = str(identity_path)

        client.connect(**connection_kwargs)
        return client

    def query_gpu_utilization(self) -> dict[int, int]:
        """Queries the core execution profile metrics of all present GPUs.

        Returns:
            dict[int, int]: A map of GPU Index entries to core load metrics.
        """
        gpu_map: dict[int, int] = {}
        command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        client = self._establish_client()

        try:
            _, stdout, _ = client.exec_command(command)
            lines = stdout.read().decode().strip().split("\n")
            for idx, load_str in enumerate(lines):
                if load_str.strip().isdigit():
                    gpu_map[idx] = int(load_str.strip())
        except Exception:
            pass
        finally:
            client.close()

        return gpu_map

    def query_active_gpu_pids(self, gpu_index: int) -> set[int]:
        """Retrieves process identifiers consuming resources on specified GPU.

        Args:
            gpu_index: Targeted hardware slot layout identifier.

        Returns:
            set[int]: A set of process identifiers captured on the physical GPU.
        """
        pids: set[int] = set()
        command = f"nvidia-smi --id={gpu_index} --query-compute-apps=pid --format=csv,noheader"
        client = self._establish_client()

        try:
            _, stdout, _ = client.exec_command(command)
            lines = stdout.read().decode().strip().split("\n")
            for line in lines:
                clean_line = line.strip()
                if clean_line.isdigit():
                    pids.add(int(clean_line))
        except Exception:
            pass
        finally:
            client.close()

        return pids

    def launch_background_job(self, command: str) -> int | None:
        """Launches detached execution contexts returning the system process ID.

        Args:
            command: Fully qualified shell operational execution string.

        Returns:
            int | None: The designated structural system PID integer value.
        """
        client = self._establish_client()
        wrapped_command = f"nohup {command} > /dev/null 2>&1 & echo $!"
        try:
            _, stdout, _ = client.exec_command(wrapped_command)
            output = stdout.read().decode().strip()
            if output.isdigit():
                return int(output)
        except Exception:
            return None
        finally:
            client.close()
        return None

    def terminate_remote_process(self, pid: int) -> None:
        """Dispatches terminal kill signals to clear resource contexts safely.

        Args:
            pid: Core operating system target workload identifier.
        """
        client = self._establish_client()
        try:
            client.exec_command(f"kill -15 {pid}")
        except Exception:
            pass
        finally:
            client.close()
