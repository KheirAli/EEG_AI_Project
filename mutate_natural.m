function y = mutate_natural(y, p,si)
    [N,M] = size(y);
    inds = rand(N,M) < p;
    values = randi(20, N, M) - 10;
    y = y + inds.* values;
    y(y>si) = si;
    y(y<1) = 1;
end