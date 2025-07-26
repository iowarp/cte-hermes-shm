from jarvis_cd.basic.pkg import Application
from jarvis_util.shell.exec import Exec
from jarvis_util.shell.pssh_exec import PsshExecInfo
from jarvis_util import *
import os
import re

def parse_size_string(size_str):
    """
    Parse human-readable size strings like '1Mb', '1kb', '1GB', etc.
    Returns size in bytes.
    """
    if isinstance(size_str, int):
        return size_str
    
    if not isinstance(size_str, str):
        return size_str
    
    # Remove any whitespace
    size_str = size_str.strip()
    
    # Pattern to match number + unit (case insensitive)
    # Only match if there's actually a unit present
    pattern = r'^(\d+(?:\.\d+)?)\s*([kmg]b?|[kmg])$'
    match = re.match(pattern, size_str.lower())
    
    if not match:
        # If no unit found, assume it's already in bytes
        try:
            return int(size_str)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")
    
    number = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    multipliers = {
        'b': 1,
        'kb': 1024,
        'mb': 1024 * 1024,
        'gb': 1024 * 1024 * 1024,
        'k': 1024,
        'm': 1024 * 1024,
        'g': 1024 * 1024 * 1024
    }
    
    if unit in multipliers:
        return int(number * multipliers[unit])
    else:
        raise ValueError(f"Unknown unit: {unit}")

class LightbeamDistributedTest(Application):
    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'hostfile', 'msg': 'Path to hostfile', 'type': str, 'default': None},
            {'name': 'transport', 'msg': 'Transport (zeromq or thallium)', 'type': str, 'default': 'thallium'},
            {'name': 'protocol', 'msg': 'Protocol (tcp or ofi+sockets)', 'type': str, 'default': 'ofi+sockets'},
            {'name': 'domain', 'msg': 'Domain (e.g., localhost)', 'type': str, 'default': 'localhost'},
            {'name': 'port', 'msg': 'Port (e.g., 8200)', 'type': int, 'default': 8200},
            {'name': 'num_msgs', 'msg': 'Number of messages per node', 'type': int, 'default': 10},
            {'name': 'msg_size', 'msg': 'Size of each message (e.g., 32, 1kb, 1Mb)', 'type': str, 'default': '32'},
        ]

    def _configure(self, **kwargs):
        self.update_config(kwargs, rebuild=False)
        self.hostfile = self.config['hostfile']
        self.transport = self.config['transport']
        self.protocol = self.config['protocol']
        self.domain = self.config['domain']
        self.port = self.config['port']
        
        # Parse msg_size if it's a string
        if 'msg_size' in self.config:
            self.config['msg_size'] = parse_size_string(self.config['msg_size'])

    def start(self):
        self.binary = 'distributed_lightbeam_test'
        # Replace None or empty string with '' for all parameters before use
        def quote_if_empty(val):
            return "''" if val is None or (isinstance(val, str) and val.strip() == '') else str(val)
        self.transport = quote_if_empty(self.config['transport'])
        self.hostfile = quote_if_empty(self.config['hostfile'])
        self.protocol = quote_if_empty(self.config['protocol'])
        self.domain = quote_if_empty(self.config['domain'])
        self.port = quote_if_empty(self.config['port'])
        cmd = f"{self.binary} {self.transport} {self.hostfile} {self.protocol} {self.domain} {self.port}"
        # Add num_msgs and msg_size if present in config
        num_msgs = self.config.get('num_msgs')
        msg_size = self.config.get('msg_size')
        if num_msgs is not None:
            cmd += f" {num_msgs}"
        if msg_size is not None:
            cmd += f" {msg_size}"
        print(f"[LightbeamDistributedTest] Launching: {cmd}")
        hosts = Hostfile(path=self.config['hostfile'])
        print(f"[LightbeamDistributedTest] Hosts: {hosts} , len(hosts): {len(hosts)}", hosts.path)
        print(f" cmd: {cmd}")
        # hosts = self.jarvis.hostfile
        Exec(cmd, MpiExecInfo(hosts= hosts, env=self.env, exec_async=False, nprocs=len(hosts), ppn=2))

    def stop(self):
        """
        Stop a running application. E.g., terminate servers, clients, etc.
        """
        pass

    def clean(self):
        """
        Destroy all data for the application. E.g., delete metadata, data directories, etc.
        """
        pass 