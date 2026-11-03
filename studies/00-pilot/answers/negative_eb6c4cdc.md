Reasoning: The output layer is purely linear: v = h2 @ A3 + c3, with no nonlinearity after it (only clipping to [0, 1] at render time). Inverting the grey values means replacing v by 1 - v everywhere. Since 1 - v = 1 - (h2 @ A3 + c3) = h2 @ (-A3) + (1 - c3), the transformation is realized exactly by negating every entry of A3 and replacing the bias c3 by 1 - c3. The hidden layers (A1, c1, A2, c2) compute the same features h2 for every coordinate, so they must not change; the picture's geometry is untouched and only the mapping from features to grey value is flipped. With c3 = 0.1014, the new bias is 1 - 0.1014 = 0.8986. Clipping commutes with this inversion (clip(1 - v) = 1 - clip(v) on [0, 1]), so the rendered picture is exactly the inverted original.

```edits
A3[:, 0] *= -1        # negate all 32 output weights
c3[0] = 0.8986        # 1 - 0.1014
```

The two statements negate the read-out weights A3 and set the output bias to 1 minus its old value, so the network now computes exactly 1 - v at every pixel. Everything upstream (A1, c1, A2, c2) is unchanged, so the spatial content of the picture is identical; only the grey scale is flipped, black to white and white to black.
