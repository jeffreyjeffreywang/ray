from typing import Any, Dict, List, Optional, Tuple

from ray.util.annotations import DeveloperAPI


@DeveloperAPI
class CommandRunnerInterface:
    """Interface to run commands on a remote cluster node.

    **Important**: This is an INTERNAL API that is only exposed for the purpose
    of implementing custom node providers. It is not allowed to call into
    CommandRunner methods from any Ray package outside the autoscaler, only to
    define new implementations for use with the "external" node provider
    option.

    Command runner instances are returned by provider.get_command_runner()."""

    def run(
        self,
        cmd: Optional[str] = None,
        timeout: int = 120,
        exit_on_fail: bool = False,
        port_forward: Optional[List[Tuple[int, int]]] = None,
        with_output: bool = False,
        environment_variables: Optional[Dict[str, object]] = None,
        run_env: str = "auto",
        ssh_options_override_ssh_key: str = "",
        shutdown_after_run: bool = False,
    ) -> str:
        """Run the given command on the cluster node and optionally get output.

        WARNING: the cloudgateway needs arguments of "run" function to be json
            dumpable to send them over HTTP requests.

        Args:
            cmd: The command to run.
            timeout: The command timeout in seconds.
            exit_on_fail: Whether to sys exit on failure.
            port_forward: List of (local, remote) ports to forward, or
                a single tuple.
            with_output: Whether to return output.
            environment_variables (Dict[str, str | int | Dict[str, str]):
                Environment variables that `cmd` should be run with.
            run_env: Options: docker/host/auto. Used in
                DockerCommandRunner to determine the run environment.
            ssh_options_override_ssh_key: if provided, overwrites
                SSHOptions class with SSHOptions(ssh_options_override_ssh_key).
            shutdown_after_run: if provided, shutdowns down the machine
                after executing the command with `sudo shutdown -h now`.
        """
        raise NotImplementedError

    def run_rsync_up(
        self, source: str, target: str, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Rsync files up to the cluster node.

        Args:
            source: The (local) source directory or file.
            target: The (remote) destination path.
        """
        raise NotImplementedError

    def run_rsync_down(
        self, source: str, target: str, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """Rsync files down from the cluster node.

        Args:
            source: The (remote) source directory or file.
            target: The (local) destination path.
        """
        raise NotImplementedError

    def remote_shell_command_str(self) -> str:
        """Return the command the user can use to open a shell."""
        raise NotImplementedError

    def run_init(
        self, *, as_head: bool, file_mounts: Dict[str, str], sync_run_yet: bool
    ) -> Optional[bool]:
        """Used to run extra initialization commands.

        Args:
            as_head: Run as head image or worker.
            file_mounts: Files to copy to the head and worker nodes.
            sync_run_yet: Whether sync has been run yet.

        Returns:
            optional: Whether initialization is necessary.
        """
        pass
