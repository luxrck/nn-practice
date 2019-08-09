#!/bin/env python3


_DEBUG = False



class EditDistance(object):

    def __init__(self, src, target, op_costs={}):
        r"""编辑操作是一个`tuple`, 包含以下操作：
            "I":    Insert
            "D":    Delete
            "R":    Replace
            "O":    Not modified
            `op_costs`: 设置这些编辑操作的代价。其中操作"O"的代价永远为0，其它操作的代价最低为1。
        """
        self.src = src
        self.target = target
        self._DP = None

        _I = op_costs.get("I", 1)
        _D = op_costs.get("R", 1)
        _R = op_costs.get("R", 1)

        if min([_I, _D, _R]) <= 0:
            raise ValueError("Edit operations (\"I, D or R\") cost should be positive.")

        self.op_costs = {"I": _I, "D": _D, "R": _R}


    def _distance(self):
        len_s, len_t = len(self.src), len(self.target)
        S, T = self.src, self.target
        I, D, R = [self.op_costs[op] for op in "IDR"]
        DP = [[0 for _ in range(len_t + 1)] for _ in range(len_s + 1)]

        for i in range(len_t + 1):
            DP[0][i] = i * I

        for i in range(len_s + 1):
            DP[i][0] = i * D

        for r in range(1, len_s + 1):
            for c in range(1, len_t + 1):
                if S[r-1] == T[c-1]:
                    DP[r][c] = DP[r-1][c-1]
                else:
                    #DP[r][c] = min([DP[r-1][c-1], DP[r-1][c], DP[r][c-1]]) + 1
                    DP[r][c] = min([DP[r-1][c-1] + R, DP[r-1][c] + D, DP[r][c-1] + I])

        return DP


    def _ensure_distance(self):
        if not self._DP:
            self._DP = self._distance()
        return self._DP


    def _moves(self, seq):
        I, D, R = [self.op_costs[op] for op in "IDR"]
        DP = self._ensure_distance()
        #len_s, len_t = len(self.src), len(self.target)

        #def direction(p, q):
        #    if p[0] == q[0]: return "l"
        #    elif p[1] == q[1]: return "u"
        #    else: return "lu"

        cost = 0
        out = []
        for i in range(1, len(seq)):
            p, pp = seq[i], seq[i-1]
            offset_r = pp[0] - p[0]
            offset_c = pp[1] - p[1]
            if offset_r == 1 and offset_c == 0:
                op = ("D", self.src[p[0]])
                cost += D
            elif offset_r == 0 and offset_c == 1:
                op = ("I", self.target[p[1]])
                cost += I
            elif offset_r == 1 and offset_c == 1:
                if DP[p[0]][p[1]] == DP[pp[0]][pp[1]]:
                    op = ("O", self.src[p[0]])
                else:
                    op = ("R", self.src[p[0]], self.target[p[1]])
                    cost += R
            out.insert(0, op)
            #next_move = direction(pp, p)
            #if next_move == "l":
            #    cost += 1
            #    out.insert(0, (("I" if pp[0] <= pp[1] else "D"), self.target[pp[1] - 1]))
            #elif next_move == "u":
            #    #import pdb; pdb.set_trace()
            #    cost += 1
            #    out.insert(0, (("I" if pp[0] <= pp[1] else "D"), self.src[pp[0] - 1]))
            #else:
            #    op = "O"
            #    if DP[p[0]][p[1]] != DP[pp[0]][pp[1]]:
            #        op = "R"; cost += 1
            #    out.insert(0, (op, self.target[pp[1] - 1]))
        if _DEBUG:
            return cost, out, seq
        return cost, out


    def distance(self):
        return self._distance()[-1][-1]


    def route(self):
        r"""输出最短编辑操作序列"""

        rs = self.routes(k=1)
        if rs: rs = rs[0]
        return rs


    def routes(self, k=1):
        r"""输出最短的`k`个编辑操作序列"""

        DP = self._ensure_distance()

        def adj_nodes(DP, p):
            pr, pc = p
            l = DP[pr-1][pc]
            lu= DP[pr-1][pc-1]
            u = DP[pr][pc-1]
            nodes = []
            if pr > 0 and pc >= 0:
                nodes.append((l, (pr-1, pc)))
            if pr >= 0 and pc > 0:
                nodes.append((u, (pr, pc-1)))
            if pr > 0 and pc > 0:
                # 当dp[i-1][j-1]不是最小代价时，(i-1, j-1) -> (i, j)的距离大于1。此时不能选择`LU`方向。
                if lu == min([lu, l, u]):
                    nodes.append((lu, (pr-1, pc-1)))
            nodes.sort()
            return [n[1] for n in nodes]

        #def adj_node(DP, visited, p):
        #    pr, pc = p
        #    nodes = []
        #    if pr > 0 and pc > 0:
        #        nodes.append((DP[pr-1][pc-1], (pr-1, pc-1)))
        #    elif pr > 0 and pc == 0:
        #        nodes.append((DP[pr-1][pc], (pr-1, pc)))
        #    elif pr == 0 and pc > 0:
        #        nodes.append((DP[pr][pc-1], (pr, pc-1)))
        #    nodes.sort()
        #    for _,n in nodes:
        #        nr, nc = n
        #        if visited[nr][nc] != 0:
        #            continue
        #        return n
        #    return None

        # 0: not visited, 1: marked, 2: visited
        #visited = [[0 for _ in range(len(DP[0]))] for _ in range(len(DP))]
        S = (len(DP) - 1, len(DP[0]) - 1)

        def _routes(DP, p, seqs, current_seq, k):
            if len(seqs) >= k: return False
            if p[0] == p[1] and p[0] == 0:
                seqs.append(current_seq); return True
            for n in adj_nodes(DP, p):
                c_seq = current_seq + [n]
                if not _routes(DP, n, seqs, c_seq, k):
                    return False
            return True

        seqs = []
        _routes(DP, S, seqs, [S], k)

        return sorted([self._moves(seq) for seq in seqs])

        #import pdb; pdb.set_trace()
        #while S:
        #    pr, pc = S[-1]
        #    if pr == pc and pr == 0:
        #        yield self._moves(S); S.pop(-1); continue
        #    if visited[pr][pc] == 0:
        #        visited[pr][pc] = 1
        #    elif visited[pr][pc] == 2:
        #        import pdb; pdb.set_trace()
        #    node = adj_node(DP, visited, (pr, pc))
        #    if not node:
        #        visited[pr][pc] = 2
        #    else:
        #        S.append(node)



#def edit_distance(w1: list, w2: list, k=1) -> (int, list):
#    lw1= len(w1); lw2 = len(w2)
#    dp = [[0 for _ in range(lw2+1)] for _ in range(lw1+1)]
#
#    for j in range(lw2+1):
#        dp[0][j] = j
#    for j in range(lw1+1):
#        dp[j][0] = j
#
#    for i in range(1, len(w1)+1):3, [('I', '他'), ('R', '我', '不'), ('O', '喜欢'), ('O', '北京')])
(3, [('R', '我', '他'), ('I', '不'), ('O', '喜欢'), ('O', '北京')])
(4, [('I', '他'), ('I', '不'), ('D', '我'), ('O', '喜欢'), ('O', '北京')])
#        for j in range(1, len(w2)+1):
#            if w1[i-1] == w2[j-1]:
#                dp[i][j] = dp[i-1][j-1]
#            else:
#                dp[i][j] = min([dp[i-1][j-1], dp[i-1][j], dp[i][j-1]]) + 1
#
#    r = len(dp)-1
#    c = len(dp[0])-1
#    op = []
#
#    while r > 0 and c > 0:
#        l = dp[r][c-1]
#        lu= dp[r-1][c-1]
#        u = dp[r-1][c]
#        m = min([l, lu, u])
#        if lu == m:
#            c -= 1; r -= 1
#            if lu == dp[r+1][c+1]:
#                op.insert(0, "O:"+w2[c])
#            else:
#                op.insert(0, "R:"+w2[c])
#        elif l == m:
#            c -= 1
#            if lw1 < lw2:3, [('I', '他'), ('R', '我', '不'), ('O', '喜欢'), ('O', '北京')])
(3, [('R', '我', '他'), ('I', '不'), ('O', '喜欢'), ('O', '北京')])
(4, [('I', '他'), ('I', '不'), ('D', '我'), ('O', '喜欢'), ('O', '北京')])
#                op.insert(0, "I:"+w2[c])
#            else:
#                op.insert(0, "D:"+w2[c])
#        elif u == m:
#            r -= 1
#            if lw1 < lw2:
#                op.insert(0, "I:"+w1[r])
#            else:
#                op.insert(0, "D:"+w1[r])
#    if r == 0 and c > 0:
#        for i in range(c, 0, -1):
#            if lw1 < lw2:
#                op.insert(0, "I:"+w2[i-1])
#            else:
#                op.insert(0, "D:"+w2[i-1])
#    if c == 0 and r > 0:
#        for i in range(r, 0, -1):
#            if lw1 < lw2:
#                op.insert(0, "I:"+w1[i-1])
#            else:
#                op.insert(0, "D:"+w1[i-1])
#    print(op, w1, w2)
#    print([0] + list(w2))
#    for i, r in enumerate(dp):
#        print(r)
#    return dp[-1][-1], op



if __name__ =="__main__":
    s = [u'我', u'喜欢', u'北京']
    #s = "horse"
    t = [u'他', u'不', u'喜欢', u'苦瓜']
    t = [u'他', u'喜欢', u'苦瓜']
    t = [u'他', u'不', u'喜欢', u'北京']
    #t = "ros"

    ed = EditDistance(s, t, op_costs={"I":1, "D":1, "R":2})

    #ed._ensure_distance()
    #r = ed._moves([(3, 4), (2, 4), (1, 4), (0, 4), (0, 3), (0, 2),(0,1),(0,0)])
    #print(r)
    print(ed.src, ed.target)
    print(ed.distance())
    gr = ed.routes(k=3)
    print(len(gr))
    for r in ed._DP:
        print(r)
    for r in gr:
        print(r)