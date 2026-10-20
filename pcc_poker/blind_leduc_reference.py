"""Independent small Leduc rule evaluator for compatibility cross-checks.

Uses physical-card chance triples, separate dictionary states, and no engine
transition, utility, legal-action, or game-tree functions.
"""
from collections import Counter
from itertools import permutations


def legal(s):
    if s['done']: return ()
    due=max(s['street_paid'])-s['street_paid'][s['turn']]
    if due==0: return ('check','bet')
    return ('fold','call','raise') if s['raises']<2 else ('fold','call')


def start(cards):
    return dict(cards=tuple(cards[:2]),board=None,next_board=cards[2],street=0,
                turn=0,paid=(1,1),street_paid=(0,0),checks=0,raises=0,
                history=(),done=False,folded=None)


def key(s):
    a=s['turn'];due=max(s['street_paid'])-s['street_paid'][a]
    board='-' if s['board'] is None else str(s['board'])
    return (a,f"r{s['street']}|h{s['cards'][a]}|b{board}|a{a}|tc{due}|{'/'.join(s['history'])}")


def step(s,action):
    if action not in legal(s): raise ValueError('illegal action')
    t=dict(s);a=s['turn'];other=1-a
    t['history']=s['history']+(action,)
    due=max(s['street_paid'])-s['street_paid'][a]
    size=(2,4)[s['street']]
    close=False
    if action=='fold': t.update(done=True,folded=a)
    elif action=='check':
        t.update(turn=other,checks=s['checks']+1);close=t['checks']==2
    else:
        amount=due if action=='call' else size+due
        paid=list(s['paid']);street_paid=list(s['street_paid'])
        paid[a]+=amount;street_paid[a]+=amount
        t.update(paid=tuple(paid),street_paid=tuple(street_paid))
        if action=='call': close=True
        else:
            t.update(turn=other,checks=0)
            if action=='raise':t['raises']+=1
    if close:
        if s['street']==1:t['done']=True
        else:t.update(board=s['next_board'],street=1,turn=0,street_paid=(0,0),checks=0,raises=0,history=t['history']+('/',))
    return t


def payoff(s,player):
    if not s['done']:raise ValueError('nonterminal')
    if s['folded'] is not None: winner=1-s['folded']
    else:
        ranks=[(card==s['board'],card) for card in s['cards']]
        winner=None if ranks[0]==ranks[1] else int(ranks[1]>ranks[0])
    pot=sum(s['paid'])
    return pot/2-s['paid'][player] if winner is None else pot-s['paid'][player] if player==winner else -s['paid'][player]


def physical_triples():
    # Six separately labeled cards, draw three without replacement: 120 deals.
    return Counter(tuple(card//2 for card in cards) for cards in permutations(range(6),3))


def profile_value(policy,player=0):
    def walk(s):
        acts=legal(s)
        if not acts:return payoff(s,player)
        return sum(float(p)*walk(step(s,a)) for p,a in zip(policy[key(s)],acts))
    return sum(count*walk(start(cards)) for cards,count in physical_triples().items())/120


def verify_tree(solver):
    checked=0
    for root in solver.roots:
        original=solver.nodes[root].state
        cards=original.private+(original.deck[0],)
        def walk(i,s):
            nonlocal checked
            n=solver.nodes[i];t=n.state
            assert legal(s)==n.actions
            assert s['cards']==t.private and s['board']==t.public
            assert s['street']==t.round_index and s['turn']==t.actor
            assert s['paid']==t.contributions and s['street_paid']==t.round_contributions
            assert s['raises']==t.raises and s['checks']==t.checks
            assert s['history']==t.history and s['done']==t.terminal and s['folded']==t.folded
            if n.key is None:
                from .engine import utility
                assert payoff(s,0)==utility(t,0) and payoff(s,1)==utility(t,1)
            else:
                assert key(s)==n.key
                for a,c in zip(legal(s),n.children):walk(c,step(s,a))
            checked+=1
        walk(root,start(cards))
    from .engine import all_deals
    rank_counts=Counter(d[:3] for d in all_deals())
    physical=physical_triples()
    assert set(rank_counts)==set(physical)
    assert all(abs(rank_counts[k]/90-physical[k]/120)<1e-14 for k in physical)
    return dict(nodes_checked=checked,physical_chance_triples=120,rank_triples=len(physical),
                uniform_rank_decks=90,all_transitions_information_keys_and_payoffs_agree=True)
