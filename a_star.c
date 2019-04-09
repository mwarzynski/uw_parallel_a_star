
struct Node {
} typedef Node;

void ga(Node* s, Node* t, int k) {
    queue[k] Q;
    queue H;

    Q[0].push(s);
    Node *m = NULL;

    while (!Q.empty()) {
        queue S;

        // Expand nodes in current queue.
        // for in parallel
        for (int i = 0; i < k; i++) {
            if (Q[i].empty()) {
                continue;
            }

            q = Q[i].extract();
            if (q.node == t) {
                if (m == NULL || f(q) < f(m)) {
                    m = q;
                }
            }

            S.push(expand(q));
        }

        // If after expand we didn't find any better path,
        // we got the final solution.
        if (m != NULL && f(m) <= min(Q, f(q))) {
            return m;
        }

        queue T = S;

        // Delete expanded nodes that have worse score.
        // for in parallel
        for (int i = 0; i < S.length(); i++) {
            if (S[i].node not in H) {
                continue;
            }
            if (H[S[i].node].g < S[i].g) {
                T[i] = NULL;
            }
        }

        // Add expanded nodes to queue (the only ones that are worth it).
        // for in parallel
        for (int i = 0; i < T.length(); i++) {
            Node *t = T[i];
            t.f = f(t);
            rand = rand_int(0, k);
            Q[rand].push(t);
            H[t.node] = t;
        }
    }
}
