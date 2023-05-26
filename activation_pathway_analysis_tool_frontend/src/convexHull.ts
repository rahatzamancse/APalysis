// Convex Hull
// http://bl.ocks.org/hollasch/9d3c098022f5524220bd84aae7623478

import { line } from "d3"
import { curveCatmullRomClosed } from "d3";

// Point/Vector Operations
type Point = [number, number]

export const vecFrom = function (p0: Point, p1: Point): Point {               // Vector from p0 to p1
    return [p1[0] - p0[0], p1[1] - p0[1]];
}

export const vecScale = function (v: Point, scale: number): Point {            // Vector v scaled by 'scale'
    return [scale * v[0], scale * v[1]];
}

export const vecSum = function (pv1: Point, pv2: Point): Point {              // The sum of two points/vectors
    return [pv1[0] + pv2[0], pv1[1] + pv2[1]];
}

export const vecUnit = function (v: Point): Point {                    // Vector with direction of v and length 1
    const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
    return vecScale(v, 1 / norm);
}

export const vecScaleTo = function (v: Point, length: number): Point {         // Vector with direction of v with specified length
    return vecScale(vecUnit(v), length);
}

export const unitNormal = function (pv0: Point, p1: Point|null=null) {           // Unit normal to vector pv0, or line segment from p0 to p1
    if (p1 != null) pv0 = vecFrom(pv0, p1);
    const normalVec: Point = [-pv0[1], pv0[0]];
    return vecUnit(normalVec);
};

// Hull Generators

const lineFn = line<{p: Point, v: Point}>()
    .curve(curveCatmullRomClosed)
    // @ts-ignore
    .x(d => d.p[0])
    // @ts-ignore
    .y(d => d.p[1]);


const smoothHull = function (polyPoints: Point[]|null, hullPadding: number) {
    // Returns the SVG path data string representing the polygon, expanded and smoothed.

    if (!polyPoints || polyPoints.length < 1) return "";
    const pointCount = polyPoints.length;

    // Handle special cases
    if (pointCount === 1) return smoothHull1(polyPoints, hullPadding);
    if (pointCount === 2) return smoothHull2(polyPoints, hullPadding);

    const hullPoints = polyPoints.map((point, index) => {
        const pNext = polyPoints[(index + 1) % pointCount];
        return {
            p: point,
            v: vecUnit(vecFrom(point, pNext))
        };
    });

    // Compute the expanded hull points, and the nearest prior control point for each.
    for (let i = 0; i < hullPoints.length; ++i) {
        const priorIndex = (i > 0) ? (i - 1) : (pointCount - 1);
        const extensionVec = vecUnit(vecSum(hullPoints[priorIndex].v, vecScale(hullPoints[i].v, -1)));
        hullPoints[i].p = vecSum(hullPoints[i].p, vecScale(extensionVec, hullPadding));
    }

    const res = lineFn(hullPoints);
    return res?res:""
}


const smoothHull1 = function (polyPoints: Point[], hullPadding: number) {
    // Returns the path for a circular hull around a single point.

    const p1 = [polyPoints[0][0], polyPoints[0][1] - hullPadding];
    const p2 = [polyPoints[0][0], polyPoints[0][1] + hullPadding];

    return 'M ' + p1
        + ' A ' + [hullPadding, hullPadding, '0,0,0', p2].join(',')
        + ' A ' + [hullPadding, hullPadding, '0,0,0', p1].join(',');
};


const smoothHull2 = function (polyPoints: Point[], hullPadding: number) {
    // Returns the path for a rounded hull around two points.

    const v = vecFrom(polyPoints[0], polyPoints[1]);
    const extensionVec = vecScaleTo(v, hullPadding);

    const extension0 = vecSum(polyPoints[0], vecScale(extensionVec, -1));
    const extension1 = vecSum(polyPoints[1], extensionVec);

    const tangentHalfLength = 1.2 * hullPadding;
    const controlDelta = vecScaleTo(unitNormal(v), tangentHalfLength);
    const invControlDelta = vecScale(controlDelta, -1);

    const control0 = vecSum(extension0, invControlDelta);
    const control1 = vecSum(extension1, invControlDelta);
    const control3 = vecSum(extension0, controlDelta);

    return 'M ' + extension0
        + ' C ' + [control0, control1, extension1].join(',')
        + ' S ' + [control3, extension0].join(',')
        + ' Z';
};

export default smoothHull;