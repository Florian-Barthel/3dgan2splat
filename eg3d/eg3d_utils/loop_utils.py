

def j_steps_until_density(i, j, m, l):
    # Calculate remaining iterations of i until the next step() call
    i_steps_left = l - (i % l) if i % l != 0 else 0

    # Calculate total j steps left
    # If i_steps_left is 0, we are at an iteration where step() will be called,
    # so we only need to consider the remaining j steps in the current i iteration.
    # Otherwise, we add the full cycles of m for each remaining i iteration,
    # and subtract the current position of j since it's already progressed.
    if i_steps_left == 0:
        # If step() is being called at this iteration of i, return the remaining j steps in this cycle of i.
        j_steps_left = m - j - 1
    else:
        # If step() is not being called at this iteration, calculate the j steps until the next call.
        # This includes the rest of the current i cycle, and all j steps in the upcoming i cycles,
        # until the i cycle where step() will be called.
        j_steps_left = (i_steps_left - 1) * m + (m - j - 1)

    return j_steps_left