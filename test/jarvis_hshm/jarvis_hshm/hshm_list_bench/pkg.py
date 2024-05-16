"""
This module provides classes and methods to launch the LabstorIpcTest application.
LabstorIpcTest is ....
"""
from jarvis_cd.basic.pkg import Application
from jarvis_util import *


class HshmListBench(Application):
    """
    This class provides methods to launch the LabstorIpcTest application.
    """
    def _init(self):
        """
        Initialize paths
        """
        pass

    def _configure_menu(self):
        """
        Create a CLI menu for the configurator method.
        For thorough documentation of these parameters, view:
        https://github.com/scs-lab/jarvis-util/wiki/3.-Argument-Parsing

        :return: List(dict)
        """
        return [
            {
                'name': 'alloc',
                'msg': 'Allocator type to use',
                'type': str,
                'choices': ['malloc', 'boost', 'scalable'],
                'default': 1,
            },
            {
                'name': 'container',
                'msg': 'Main data structure',
                'type': str,
                'choices': ['list', 'slist'],
                'default': 'list',
            },
            {
                'name': 'subtype',
                'msg': 'Type stored in container',
                'type': str,
                'choices': ['size_t', 'std::string', 'string'],
                'default': 'size_t',
            },
            {
                'name': 'test_case',
                'msg': 'Test case to run',
                'type': str,
                'choices': ['allocate', 'emplace', 'forward_iterator',
                            'copy', 'move'],
                'default': 'size_t',
            },
            {
                'name': 'ops',
                'msg': 'The # operations to generate per node',
                'type': str,
                'default': '4k',
            },
        ]

    def _configure(self, **kwargs):
        """
        Converts the Jarvis configuration to application-specific configuration.
        E.g., OrangeFS produces an orangefs.xml file.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        pass

    def start(self):
        """
        Launch an application. E.g., OrangeFS will launch the servers, clients,
        and metadata services on all necessary pkgs.

        :return: None
        """
        cmd = [
            'hshm_bench_list',
            self.config['alloc'],
            self.config['container'],
            self.config['subtype'],
            self.config['test_case'],
            self.config['ops'],
        ]
        cmd = ' '.join(cmd)
        Exec(cmd,
             LocalExecInfo(env=self.env,
                           do_dbg=self.config['do_dbg'],
                           dbg_port=self.config['dbg_port']))

    def stop(self):
        """
        Stop a running application. E.g., OrangeFS will terminate the servers,
        clients, and metadata services.

        :return: None
        """
        pass

    def clean(self):
        """
        Destroy all data for an application. E.g., OrangeFS will delete all
        metadata and data directories in addition to the orangefs.xml file.

        :return: None
        """
        pass
