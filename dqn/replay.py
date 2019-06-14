import retro


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
    # run = "Prio7"
    # numbers = ["057", "081", "104", "105"]
    # run = "Prio8"
    # numbers = ["019", "020", "023", "024", "050", "058", "060", "087", "088", "092", "107"]
    # run = "Prio9"
    # numbers = [150, 477, 264, 445, 199, 379, 225]
    run = "V2"
    numbers = [268]

    # recording = RECORDING_NAME
    for number in numbers:
        recording = './recordings/{}/SuperMarioBros3-Nes-1Player.World1.Level1-000{}.bk2'.format(run, str(number))
        replay(recording)

    # to convert the movie .bk2 into a mp4 video, run the following on the command line:
    # python -m retro.scripts.playback_movie path_to_recording
    # where path_to_recording can be replaced with the actual path (e.g., ./recordings/SuperMarioBros3-Nes-1Player.World1.Level1-000000.bk2)
