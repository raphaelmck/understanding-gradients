> The surface generates, and slowly spins

Suppose you have a landscape, or terrain, and you want to find which direction to go, in order to climb up, or ascend, as fast as possible. 

> Surface stops spinning and the axis of the graph appear.

In three dimentions, you can think of a function $f(x,y)$ that turns two numbers into a height. 

> A dot on the xy plane, dotted line up to the surface and another dot with a value

Pick a point on the ground, and the function tells you how high you are. Now what we want is local steepest ascent: at one point, there isn't a single slope, like you would see in a single variable derivative

> Possibly show something about the "normal" derivative

There are instead infinitely many slopes, one for each direction you could choose to walk

> A couple arrows on the tangent plane, small numbers +0.3, -0.1 ...

We then introduce a way to organize all these directional slopes into something simpler

> Arrow fade into the special arrow

Notice the change you experience doesn't depend of the direction itself, it depend on how aligned that direction is, with the uphill tendency of the surface

> One direction arrow rotating, snall number grows and shrinks

When you move with this direction, the height increases quickly, when you move against it, the height drops quickly, when you move perpendicular to it,

> Perpendicular arrow shows 0

there's essentially no change at all

> $\nabla f$ appears

This single vector encodes all local slopes at once. For any direction you choose, the rate of change is determined by how much that direction overlaps with this vector. Ans when the two directions line up perfectly, the increase is as large as it can possibly be.

> transition into level curves

As a consequence, directions that stay on the same height, level curves, must be perpendicular to the gradient, They're directions of zero overlap

> Show dot product

This is why they show up in physics, numerical methods, and learning algorithms

> Cycle through visualizations of those

So for a function $f(x,y)$, $\frac{\partial f}{\partial x}$ tells you, if I move only in the $x$ direction, how fast does the height change, similarly for $\frac{\partial f}{\partial y}$, these are direction-specific measurements.

Now take any direction in the plane, representend by a unit vector $\mathbf{v}=(v_x,v_y)$, moving in direction $\mathbf{v}$ means move $v_x$ in the $x$-direction, and
