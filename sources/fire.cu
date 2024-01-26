// Ported from GLSL shader made by Ian McEwan, Ashima Arts (https://www.shadertoy.com/view/MlKSWm)
// 

#ifndef TERMSHADER_FIRE_CU_


using namespace ts;


__device__ static v3f mod289(const v3f& x) {

	return x - floor_d(x/289.0f)*289.0f; 
}


__device__ static v4f mod289(const v4f& x) {

	return x - floor_d(x/289.0f)*289.0f;
}


__device__ static v4f permute(const v4f& x) {

	return mod289((x*34.0f+1.0f)*x);
}

__device__ static v4f taylorInvSqrt(const v4f& r) {

	return 1.79284291400159 - 0.85373472095314 * r;
}


__device__ static f32 snoise(const v3f& v) {

	const v2f C = vec2_d(1.0f/6.0f, 1.0f/3.0f) ;
	const v4f D = vec4_d(0.0f, 0.5f, 1.0f, 2.0f);

    // First corner
	v3f i  = floor_d(v + dot_d(v, vec3_d(C.y, C.y, C.y)));
	v3f x0 = v - i + dot_d(i, vec3_d(C.x, C.x, C.x));

    // Other corners
	v3f g = step_d(vec3_d(x0.y, x0.z, x0.x), x0);
	v3f l = 1.0f - g;
	v3f i1 = min_d( g, vec3_d(l.z, l.x, l.y) );
	v3f i2 = max_d( g, vec3_d(l.z, l.x, l.y) );

	v3f x1 = x0 - i1 + vec3_d(C.x, C.x, C.x);
	v3f x2 = x0 - i2 + vec3_d(C.y, C.y, C.y); // 2.0*C.x = 1/3 = C.y
	v3f x3 = x0 - vec3_d(D.y, D.y, D.y);	  // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
	i = mod289(i);
	v4f p = permute( permute( permute(
					   i.z + vec4_d(0.0f, i1.z, i2.z, 1.0f))
					 + i.y + vec4_d(0.0f, i1.y, i2.y, 1.0f))
					 + i.x + vec4_d(0.0f, i1.x, i2.x, 1.0f));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	f32 n_ = 0.142857142857f; // 1.0/7.0
	v3f	ns = n_ * vec3_d(D.w, D.y, D.z) - vec3_d(D.x, D.z, D.x);

	v4f j = p - 49.0f * floor_d(p * ns.z * ns.z);	//	mod(p,7*7)

	v4f x_ = floor_d(j * ns.z);
	v4f y_ = floor_d(j - 7.0f * x_ );		// mod(j,N)

	v4f x = x_ * ns.x + vec4_d(ns.y, ns.y, ns.y, ns.y);
	v4f y = y_ * ns.x + vec4_d(ns.y, ns.y, ns.y, ns.y);
	v4f h = 1.0f - abs_d(x) - abs_d(y);

	v4f b0 = vec4_d( x.x, x.y, y.x, y.y );
	v4f b1 = vec4_d( x.z, x.w, y.z, y.w );

	v4f s0 = floor_d(b0)*2.0f + 1.0f;
	v4f s1 = floor_d(b1)*2.0f + 1.0f;
	v4f sh = -step_d(h, vec4_d(0.0f, 0.0f, 0.0f, 0.0f));

	v4f a0 = vec4_d(b0.x, b0.z, b0.y, b0.w) + vec4_d(s0.x, s0.z, s0.y, s0.w)*vec4_d(sh.x, sh.x, sh.y, sh.y);
	v4f a1 = vec4_d(b1.x, b1.z, b1.y, b1.w) + vec4_d(s1.x, s1.z, s1.y, s1.w)*vec4_d(sh.z, sh.z, sh.w, sh.w);

	v3f p0 = vec3_d(a0.x, a0.y, h.x);
	v3f p1 = vec3_d(a0.z, a0.w, h.y);
	v3f p2 = vec3_d(a1.x, a1.y, h.z);
	v3f p3 = vec3_d(a1.z, a1.w, h.w);

    //Normalise gradients
	v4f norm = rsqrt_d(vec4_d(dot_d(p0,p0), dot_d(p1,p1), dot_d(p2, p2), dot_d(p3,p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

    // Mix final noise value
	v4f m = max_d(0.6f - vec4_d(dot_d(x0,x0), dot_d(x1,x1), dot_d(x2,x2), dot_d(x3,x3)), vec4_d(0.0f, 0.0f, 0.0f, 0.0f));
	m = m * m;
	return 42.0f * dot_d( m*m, vec4_d(dot_d(p0,x0), dot_d(p1,x1), dot_d(p2,x2), dot_d(p3,x3)) );
}


__device__ static f32 prng(const v2f& seed) {

	v2f sd = fract_d(seed * vec2_d(5.3983f, 5.4427));
	sd += dot_d(vec2_d(sd.y, sd.x), sd + vec2_d(21.5351f, 14.3137f));
	return fract_d(sd.x * sd.y * 95.4337f);
}


__device__ static f32 noiseStack(v3f pos, int octaves, f32 falloff) {
	
	f32 noise = snoise(pos);
	f32 off = 1.0f;

	if (octaves > 1) {
		pos *= 2.0f;
		off *= falloff;
		noise = (1.0f-off)*noise + off*snoise(pos);
	}

	if (octaves > 2) {
		pos *= 2.0f;
		off *= falloff;
		noise = (1.0f-off)*noise + off*snoise(pos);
	}
	if (octaves > 3) {
		pos *= 2.0f;
		off *= falloff;
		noise = (1.0f-off)*noise + off*snoise(pos);
	}

	return (1.0f+noise)/2.0f;
}


__device__ static v2f noiseStackUV(const v3f& pos, int octaves, f32 falloff, f32 diff) {

	f32 displaceA = noiseStack(pos, octaves, falloff);
	f32 displaceB = noiseStack(pos+vec3_d(3984.293f, 423.21f, 5235.19f), octaves, falloff);
	return vec2_d(displaceA, displaceB);
}


__device__ void shade_fire(v4f& fragColor, const v2f& fragCoord, f32 time, const v2f& resolution) {

	if (fragCoord.y < 1) {
		fragColor = vec4_d(0.0f, 0.0f, 0.0f, 1.0f);
		return;
	}

	v2f drag = vec2_d(0.0f, 0.0f);
	v2f offset = vec2_d(0.0f, 0.0f);

	f32 xpart = fragCoord.x/resolution.x;
	f32 ypart = fragCoord.y/resolution.y;
	
	f32 clip = 50.0f;
	f32 ypartClip = fragCoord.y/clip;
	f32 ypartClippedFalloff = clamp_d(2.0f - ypartClip, 0.0f, 1.0f);
	f32 ypartClipped = min(ypartClip, 1.0f);
	f32 ypartClippedn = 1.0f - ypartClipped;
	
	f32 xfuel = 1.0-abs(2.0f*xpart - 1.0f);
	
	f32 timeSpeed = 0.5f;
	f32 realTime = timeSpeed*time;
	
	v2f coordScaled = 0.01f*fragCoord - 0.02f*vec2_d(offset.x, 0.0);
	v3f position = vec3_d(coordScaled.x, coordScaled.y, 0.0f) + vec3_d(1223.0f, 6434.0f, 8425.0f);
	v3f flow = vec3_d(4.1f*(0.5f-xpart)*powf(ypartClippedn, 4.0f), -2.0f*xfuel*powf(ypartClippedn, 64.0f), 0.0f);
	v3f timing = realTime*vec3_d(0.0f, -1.7f, 1.1f) + flow;
	
	v3f displacePos = vec3_d(1.0f, 0.5f, 1.0f)*2.4f*position + realTime*vec3_d(0.01f, -0.7f, 1.3f);
	v3f displace3 = vec3_d(noiseStackUV(displacePos, 2, 0.4f, 0.1f), 0.0f);
	
	v3f noiseCoord = (vec3_d(2.0f, 1.0f, 1.0f)*position + timing + 0.4f*displace3)/1.0f;
	f32 noise = noiseStack(noiseCoord, 3, 0.4f);
	
	f32 flames = powf(ypartClipped, 0.3f*xfuel)*powf(noise, 0.3f*xfuel);
	
	f32 f = ypartClippedFalloff*pow(1.0f-flames*flames*flames, 8.0f);
	f32 fff = f*f*f;
	v3f fire = 1.5f*vec3_d(f, fff, fff*fff);
	//
	// smoke
	f32 smokeNoise = 0.5f + snoise(0.4f*position+timing*vec3_d(1.0f, 1.0f, 0.2f))/2.0f;
	v3f smoke = vec3_d(0.3f*pow(xfuel, 3.0f)*pow(ypart, 2.0f)*(smokeNoise+0.4f*(1.0f-noise)));
	//
	// sparks
	f32 sparkGridSize = 30.0f;
	v2f sparkCoord = fragCoord - vec2_d(2.0f*offset.x, 190.0f*realTime);
	sparkCoord -= 30.0f*noiseStackUV(0.01f*vec3_d(sparkCoord, 30.0f*time), 1, 0.4f, 0.1f);
	sparkCoord += 100.0f*flow.xy;
	
	if (mod_d(sparkCoord.y/sparkGridSize, 2.0f) < 1.0f) 
		sparkCoord.x += 0.5f*sparkGridSize;
	
	v2f sparkGridIndex = floor_d(sparkCoord/sparkGridSize);
	f32 sparkRandom = prng(sparkGridIndex);
	f32 sparkLife = min(10.0f*(1.0f-min((sparkGridIndex.y+(190.0f*realTime/sparkGridSize))/(24.0f-20.0f*sparkRandom), 1.0f)), 1.0f);
	v3f sparks = vec3_d(0.0f);
	if (sparkLife > 0.0f) {
		f32 sparkSize = xfuel * xfuel * sparkRandom * 0.08f;
		f32 sparkRadians = 999.0f * sparkRandom * 2.0f * M_PI + 2.0f * time;
		v2f sparkCircular = vec2_d(sin(sparkRadians), cos(sparkRadians));
		v2f sparkOffset = (0.5f-sparkSize)*sparkGridSize*sparkCircular;
		v2f sparkModulus = mod_d(sparkCoord+sparkOffset, sparkGridSize) - 0.5f*vec2_d(sparkGridSize);
		f32 sparkLength = length_d(sparkModulus);
		f32 sparksGray = max(0.0, 1.0 - sparkLength/(sparkSize*sparkGridSize));
		sparks = sparkLife*sparksGray*vec3_d(1.0f, 0.3f, 0.0f);
	}
	//
	fragColor = vec4_d(max_d(fire, sparks)+smoke, 1.0f);
}


#define TERMSHADER_FIRE_CU_
#endif

