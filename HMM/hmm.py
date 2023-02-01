#!/usr/bin/env python3
import json

def viterbi(model):
    state_space, obs_space = model["state_space"], model["obs_space"]
    start_prob = model["start_prob"]
    trans_prob = model["trans_prob"]
    emit_prob = model["emit_prob"]
    obs_seqs = model["obs_seqs"]

    T = len(obs_seqs)
    likelihood = {state: [-float('Inf') for _ in range(T)]
                  for state in state_space}
    path = {state: [None for _ in range(T)] for state in state_space}

    for state in state_space:
        likelihood[state][0] = start_prob[state] * \
                               emit_prob[state][obs_seqs[0]]

    for t in range(1, T):
        for state in state_space:
            for prev_state in state_space:
                new_likelihood = likelihood[prev_state][t-1] *\
                                 trans_prob[prev_state][state] *\
                                 emit_prob[state][obs_seqs[t]]
                if likelihood[state][t] < new_likelihood:
                    likelihood[state][t] = new_likelihood
                    path[state][t] = prev_state

    most_likely_final_state = None
    most_likely_final_prob = -1
    for state in state_space:
        if most_likely_final_prob < likelihood[state][T-1]:
            most_likely_final_prob = likelihood[state][T-1]
            most_likely_final_state = state

    hidden_seqs = [most_likely_final_state]
    curr_state = most_likely_final_state
    for t in range(T-1, 0, -1):
        hidden_seqs.append(path[curr_state][t])
        curr_state = path[curr_state][t]
    hidden_seqs.reverse()

    return likelihood, hidden_seqs


def per_model(model):
    likelihood, hidden = viterbi(model)

    # Print
    obs_seqs = model["obs_seqs"]
    index_row = [str(i-1) for i in range(len(obs_seqs) + 1)]
    header_row = ["observations"] + obs_seqs
    state_space = ["state_space"]
    max_obs_len = max([len(s) for s in header_row + state_space])

    def format_table(x):
        return " ".join([s.rjust(max_obs_len, " ") for s in x])
    print(format_table(index_row))
    print(format_table(header_row))
    for state in model["state_space"]:
        row = [state.rjust(max_obs_len, " ")]
        for t in range(len(obs_seqs)):
            s = str(round(likelihood[state][t], max_obs_len))
            row.append(s)
        print(format_table(row))

    outcome = ["best seq"] + hidden
    print(format_table(outcome))
    # print("reference: " + model["reference"])


if __name__ == '__main__':
    with open("models.json", "rb") as f:
        models = json.load(f)
        f.close()
    for name, model in models.items():
        print(("== " + name + " ").ljust(50, "="))
        per_model(model)
        print("\n")
