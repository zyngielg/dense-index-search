def remove_duplicates_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x) or x == '')]

def torch_percentile(tensor, p):
    assert p in range(1, 100+1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()

def uniq(l):
    l = remove_duplicates_preserve_order(l)
    # ignore the last id due to the way the views are created
    if 147069 in l:
        l.remove(147069)
    return l