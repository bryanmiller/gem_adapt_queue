# Matt

import random


def generate_events(grid_size, event_type, probability, event_max):
    """
    Generate up to 'number' of events at random times throughout the night.
    Return events as a list of lists containing the event type and time grid index at which the event occurs.

    Example
    -------
    >>> events = events(120, 'Target of Opportunity', 0.1, 4)
    >>> print(events)
    [['Target of Opportunity', 50], ['Target of Opportunity', 20]]

    Parameters
    ----------
    grid_size : int
        number of discrete time vales throughout observing window.

    event_type : str
        type of event ('Target of Opportunity' or 'Condition change' for sky conditions changes).

    probability : float
        probability of an event occurring.

    event_max : int
        number of potential events.
    """
    verbose = False

    nt = grid_size
    p = probability
    n = event_max

    if verbose:
        print('\nGenerating random events:')
        print('grid_size', nt)
        print('type', event_type)
        print('probability', p)
        print('number', n)

    events = []

    if p > 0.:  # if event have probability greater than zero

        for i in range(n):
            random_num = random.random()  # 'roll the dice' for ToO (number in range [0,1))
            if verbose:
                print('random_num', random_num)

            if random_num <= p:  # if roll is >= probability, generate event.
                event_grid_index = random.randint(0, nt - 1)  # random time grid index somewhere in the night.
                events.append([event_type, event_grid_index])  # save event type and time grid index.
                if verbose:
                    print('added event:', [event_type, event_grid_index])

    return events


def test_events():
    type = 'Nuclear meltdown'
    grid_size = 20
    p = 'imminent'
    p = 0.1
    max = 20
    random.seed(1000)
    events = generate_events(grid_size=grid_size, event_type=type, probability=p, event_max=max)
    print(events)
    assert events == [['Nuclear meltdown', 11], ['Nuclear meltdown', 5], ['Nuclear meltdown', 15]]
    print('Test successful!')
    return


if __name__=='__main__':
    test_events()
