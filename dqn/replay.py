import retro

from config import RECORDING_NAME


def replay(recording):
    movie = retro.Movie(recording)
    movie.step()

    env = retro.make(
        game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        env.step(keys)
        env.render()


if __name__ == "__main__":
    recording = RECORDING_NAME
    replay(recording)

    # to convert the movie .bk2 into a mp4 video, run the following on the command line:
    # python -m retro.scripts.playback_movie path_to_recording
    # where path_to_recording can be replaced with the actual path (e.g., ./recordings/SuperMarioBros3-Nes-1Player.World1.Level1-000000.bk2)
