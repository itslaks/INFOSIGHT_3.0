from flask import Flask, Blueprint, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import io
import numpy as np
from datetime import datetime
import hashlib
import cv2

trueshot_ai = Blueprint('trueshot_ai', __name__, url_prefix='/trueshot_ai')

class AdvancedImageAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._setup_model()
        self.classes = ['AI-generated', 'Real']
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Enhanced detection patterns
        self.ai_patterns = {
            'diffusion_artifacts': 0.0,
            'gan_signatures': 0.0,
            'unnatural_smoothness': 0.0,
            'pixel_repetition': 0.0,
            'color_banding': 0.0
        }

    def _setup_model(self):
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2)
        )
        try:
            model.load_state_dict(torch.load('best_model9.pth', map_location=self.device))
            model.eval()
        except:
            print("Warning: Model file not found, using untrained model")
        return model.to(self.device)

    def _analyze_metadata(self, image):
        """Enhanced metadata analysis"""
        metadata = {
            'format': str(image.format) if image.format else 'Unknown',
            'mode': str(image.mode),
            'has_exif': bool(image.getexif()),
            'exif_data': {},
            'suspicious_indicators': []
        }
        
        exif = image.getexif()
        if exif:
            metadata['exif_data'] = {
                'camera_make': str(exif.get(271, 'Unknown')),
                'camera_model': str(exif.get(272, 'Unknown')),
                'software': str(exif.get(305, 'Unknown')),
                'datetime': str(exif.get(306, 'Unknown'))
            }
            
            # Check for AI generation software signatures
            software = str(exif.get(305, '')).lower()
            ai_tools = ['midjourney', 'dall-e', 'dalle', 'stable diffusion', 'stablediffusion', 
                       'playground', 'leonardo', 'firefly', 'craiyon', 'deepai']
            if any(ai_tool in software for ai_tool in ai_tools):
                metadata['suspicious_indicators'].append('AI generation software detected in EXIF')
        else:
            metadata['suspicious_indicators'].append('No EXIF data (common in AI images)')
        
        return metadata

    def _analyze_compression_artifacts(self, image):
        """Enhanced compression analysis"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Multiple edge detection methods
        edges_canny = cv2.Canny(gray, 100, 200)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        
        edge_density = np.sum(edges_canny > 0) / edges_canny.size
        edge_variance = np.var(edges_sobel)
        
        # Block artifact detection (8x8 blocks common in JPEG)
        h, w = gray.shape
        block_scores = []
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8]
                block_scores.append(np.std(block))
        
        block_consistency = np.std(block_scores) if block_scores else 0
        
        # DCT coefficient analysis for JPEG detection
        dct_score = 0
        if h > 64 and w > 64:
            sample = gray[h//2:h//2+64, w//2:w//2+64].astype(np.float32)
            dct = cv2.dct(sample)
            dct_score = float(np.std(dct))
        
        return {
            'edge_density': float(edge_density),
            'edge_variance': float(edge_variance),
            'block_consistency': float(block_consistency),
            'dct_score': float(dct_score),
            'has_compression': bool(edge_density > 0.05),
            'natural_compression': bool(block_consistency > 5.0 and dct_score > 100)
        }

    def _analyze_frequency_domain(self, image):
        """Advanced frequency domain analysis"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        gray = np.array(image.convert('L'))
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Multiple frequency band analysis
        high_freq = magnitude.copy()
        high_freq[crow-30:crow+30, ccol-30:ccol+30] = 0
        high_freq_energy = np.sum(high_freq) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Low frequency analysis
        low_freq = np.zeros_like(magnitude)
        low_freq[crow-30:crow+30, ccol-30:ccol+30] = magnitude[crow-30:crow+30, ccol-30:ccol+30]
        low_freq_energy = np.sum(low_freq) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Mid frequency (real photos have good mid-frequency content)
        mid_freq = magnitude.copy()
        mid_freq[crow-15:crow+15, ccol-15:ccol+15] = 0
        mid_freq[crow-45:crow+45, ccol-45:ccol+45] = 0
        mid_freq_energy = np.sum(mid_freq) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # AI images often have unusual frequency distribution
        freq_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
        
        return {
            'high_freq_ratio': float(high_freq_energy),
            'low_freq_ratio': float(low_freq_energy),
            'mid_freq_ratio': float(mid_freq_energy),
            'freq_balance': float(freq_ratio),
            'spectral_anomaly': bool(high_freq_energy < 0.3 or freq_ratio < 0.1),
            'ai_signature': bool(freq_ratio < 0.12)
        }

    def _analyze_texture_consistency(self, image):
        """Enhanced texture pattern analysis"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        patch_size = 32
        
        variances = []
        means = []
        local_patterns = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = img_array[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
                means.append(np.mean(patch))
                
                # Local binary pattern approximation
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                local_patterns.append(np.std(gray_patch))
        
        texture_consistency = np.std(variances)
        pattern_uniformity = np.std(local_patterns)
        
        # AI images often have unnaturally uniform textures
        is_too_uniform = texture_consistency < 400 and pattern_uniformity < 8
        
        return {
            'texture_consistency': float(texture_consistency),
            'pattern_uniformity': float(pattern_uniformity),
            'uniform_texture': bool(is_too_uniform),
            'variance_range': float(np.max(variances) - np.min(variances)) if variances else 0.0,
            'ai_smoothness': bool(is_too_uniform and pattern_uniformity < 6)
        }

    def _analyze_color_distribution(self, image):
        """Deep color analysis with AI detection patterns"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        stat = ImageStat.Stat(image)
        img_array = np.array(image)
        
        # Histogram analysis
        hist_r = image.histogram()[0:256]
        hist_g = image.histogram()[256:512]
        hist_b = image.histogram()[512:768]
        
        def calc_entropy(hist):
            hist = np.array(hist) / sum(hist) if sum(hist) > 0 else np.array(hist)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        
        entropy = (calc_entropy(hist_r) + calc_entropy(hist_g) + calc_entropy(hist_b)) / 3
        
        # Color banding detection (common in AI images)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_diversity = unique_colors / total_pixels if total_pixels > 0 else 0
        
        # Channel correlation (AI images often have unusual correlations)
        r_channel = img_array[:,:,0].flatten()
        g_channel = img_array[:,:,1].flatten()
        b_channel = img_array[:,:,2].flatten()
        
        rg_corr = np.corrcoef(r_channel, g_channel)[0, 1]
        rb_corr = np.corrcoef(r_channel, b_channel)[0, 1]
        gb_corr = np.corrcoef(g_channel, b_channel)[0, 1]
        
        avg_correlation = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
        
        return {
            'entropy': float(entropy),
            'color_diversity': float(color_diversity),
            'unique_colors': int(unique_colors),
            'channel_correlation': float(avg_correlation),
            'color_range': [[int(x[0]), int(x[1])] for x in stat.extrema],
            'mean_colors': [float(x) for x in stat.mean],
            'std_colors': [float(x) for x in stat.stddev],
            'low_entropy': bool(entropy < 5.5),
            'suspicious_correlation': bool(avg_correlation > 0.88),
            'color_banding': bool(color_diversity < 0.25)
        }

    def _analyze_noise_patterns(self, image):
        """Advanced noise analysis for AI detection"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        gray = np.array(image.convert('L'))
        
        # Apply high-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred)
        
        noise_level = float(np.std(noise))
        noise_mean = float(np.mean(noise))
        
        # Natural images have Gaussian noise from sensors
        # AI images often have no noise or artificial patterns
        hist, bins = np.histogram(noise.flatten(), bins=50, range=(-50, 50))
        noise_distribution = np.std(hist)
        
        # Check for noise uniformity across image
        h, w = noise.shape
        quadrants = [
            noise[0:h//2, 0:w//2],
            noise[0:h//2, w//2:w],
            noise[h//2:h, 0:w//2],
            noise[h//2:h, w//2:w]
        ]
        quadrant_noise = [np.std(q) for q in quadrants]
        noise_uniformity = np.std(quadrant_noise)
        
        # Analyze noise frequency characteristics
        noise_fft = np.fft.fft2(noise)
        noise_magnitude = np.abs(noise_fft)
        noise_energy = float(np.sum(noise_magnitude))
        
        return {
            'noise_level': noise_level,
            'noise_mean': noise_mean,
            'noise_distribution': float(noise_distribution),
            'noise_uniformity': float(noise_uniformity),
            'noise_energy': noise_energy,
            'has_natural_noise': bool(5 < noise_level < 30),
            'suspiciously_clean': bool(noise_level < 3),
            'artificial_noise': bool(noise_uniformity > 2.5)
        }

    def _detect_ai_artifacts(self, image):
        """Detect specific AI generation artifacts"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        artifacts = {}
        
        # 1. Pixel repetition (copy-paste artifacts in diffusion models)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        template_size = 16
        h, w = gray.shape
        repetition_score = 0
        
        if h > template_size * 4 and w > template_size * 4:
            template = gray[h//2:h//2+template_size, w//2:w//2+template_size]
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            matches = np.where(result > 0.95)
            repetition_score = len(matches[0])
        
        artifacts['pixel_repetition'] = float(repetition_score / 100)
        
        # 2. Unnatural smoothness in faces/skin (GAN artifacts)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        artifacts['smoothness_score'] = float(laplacian_var)
        artifacts['unnatural_smoothness'] = bool(laplacian_var < 40)
        
        # 3. Grid patterns (common in diffusion models)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Look for regular grid patterns
        center = np.array(magnitude_spectrum.shape) // 2
        grid_score = 0
        for offset in [8, 16, 32, 64]:
            if center[0] + offset < magnitude_spectrum.shape[0] and center[1] + offset < magnitude_spectrum.shape[1]:
                grid_score += magnitude_spectrum[center[0] + offset, center[1]]
                grid_score += magnitude_spectrum[center[0], center[1] + offset]
        
        artifacts['grid_pattern_score'] = float(grid_score / 1000000)
        artifacts['has_grid_pattern'] = bool(grid_score > 6000000)
        
        # 4. Chromatic aberration (real cameras have it, AI often doesn't)
        r_channel = img_array[:,:,0]
        b_channel = img_array[:,:,2]
        edge_r = cv2.Canny(r_channel, 100, 200)
        edge_b = cv2.Canny(b_channel, 100, 200)
        chromatic_diff = float(np.sum(np.abs(edge_r.astype(float) - edge_b.astype(float))))
        artifacts['chromatic_aberration'] = chromatic_diff
        artifacts['has_chromatic_aberration'] = bool(chromatic_diff > 1000)
        
        return artifacts

    def _comprehensive_analysis(self, image):
        """Comprehensive multi-factor analysis"""
        width, height = image.size
        aspect_ratio = width / height
        
        if image.mode in ('RGBA', 'P'):
            rgb_image = image.convert('RGB')
        else:
            rgb_image = image
        
        # Run all enhanced analysis functions
        metadata = self._analyze_metadata(image)
        compression = self._analyze_compression_artifacts(rgb_image)
        frequency = self._analyze_frequency_domain(rgb_image)
        texture = self._analyze_texture_consistency(rgb_image)
        color = self._analyze_color_distribution(rgb_image)
        noise = self._analyze_noise_patterns(rgb_image)
        artifacts = self._detect_ai_artifacts(rgb_image)
        
        return {
            'dimensions': {
                'width': width,
                'height': height,
                'aspect_ratio': round(aspect_ratio, 2),
                'total_pixels': width * height
            },
            'metadata': metadata,
            'compression': compression,
            'frequency': frequency,
            'texture': texture,
            'color': color,
            'noise': noise,
            'artifacts': artifacts,
            'file_size_kb': 0
        }

    def _calculate_confidence_factors(self, analysis):
        """Enhanced confidence calculation with balanced weights"""
        factors = []
        ai_score = 0
        real_score = 0
        
        # Metadata factors - STRONG INDICATOR
        if not analysis['metadata']['has_exif']:
            factors.append({
                'factor': 'Missing EXIF metadata',
                'impact': 0.05,
                'direction': 'ai',
                'severity': 'low'
            })
            ai_score += 0.05
        else:
            exif_data = analysis['metadata']['exif_data']
            if exif_data.get('camera_make') != 'Unknown' or exif_data.get('camera_model') != 'Unknown':
                factors.append({
                    'factor': f"Camera EXIF data: {exif_data.get('camera_make', '')} {exif_data.get('camera_model', '')}",
                    'impact': 0.25,
                    'direction': 'real',
                    'severity': 'high'
                })
                real_score += 0.25
            else:
                factors.append({
                    'factor': 'EXIF metadata present',
                    'impact': 0.15,
                    'direction': 'real',
                    'severity': 'medium'
                })
                real_score += 0.15
        
        # Check for AI software in EXIF
        if analysis['metadata']['suspicious_indicators']:
            for indicator in analysis['metadata']['suspicious_indicators']:
                if 'AI generation software' in indicator:
                    factors.append({
                        'factor': indicator,
                        'impact': 0.30,
                        'direction': 'ai',
                        'severity': 'high'
                    })
                    ai_score += 0.30
        
        # Noise analysis - MOST RELIABLE INDICATOR FOR REAL PHOTOS
        if analysis['noise']['suspiciously_clean']:
            factors.append({
                'factor': 'Extremely low noise level (AI characteristic)',
                'impact': 0.22,
                'direction': 'ai',
                'severity': 'high'
            })
            ai_score += 0.22
        elif analysis['noise']['has_natural_noise']:
            factors.append({
                'factor': f"Natural sensor noise detected (level: {analysis['noise']['noise_level']:.2f})",
                'impact': 0.30,
                'direction': 'real',
                'severity': 'high'
            })
            real_score += 0.30
        
        # Compression artifacts - STRONG INDICATOR FOR REAL PHOTOS
        if analysis['compression']['natural_compression']:
            factors.append({
                'factor': 'Natural JPEG compression artifacts detected',
                'impact': 0.25,
                'direction': 'real',
                'severity': 'high'
            })
            real_score += 0.25
        
        # Chromatic aberration - real cameras have it
        if analysis['artifacts'].get('has_chromatic_aberration', False):
            factors.append({
                'factor': 'Chromatic aberration present (camera lens characteristic)',
                'impact': 0.18,
                'direction': 'real',
                'severity': 'medium'
            })
            real_score += 0.18
        
        # Texture factors
        if analysis['texture']['ai_smoothness']:
            factors.append({
                'factor': 'Artificial smoothness pattern detected',
                'impact': 0.15,
                'direction': 'ai',
                'severity': 'medium'
            })
            ai_score += 0.15
        elif analysis['texture']['texture_consistency'] > 1000:
            factors.append({
                'factor': f"Natural texture variation (score: {analysis['texture']['texture_consistency']:.0f})",
                'impact': 0.18,
                'direction': 'real',
                'severity': 'medium'
            })
            real_score += 0.18
        
        # Frequency domain analysis
        if analysis['frequency']['ai_signature']:
            factors.append({
                'factor': 'Abnormal frequency distribution (low high-frequency content)',
                'impact': 0.12,
                'direction': 'ai',
                'severity': 'medium'
            })
            ai_score += 0.12
        elif analysis['frequency']['freq_balance'] > 0.20:
            factors.append({
                'factor': f"Natural frequency distribution (ratio: {analysis['frequency']['freq_balance']:.3f})",
                'impact': 0.15,
                'direction': 'real',
                'severity': 'medium'
            })
            real_score += 0.15
        
        # Color analysis
        if analysis['color']['suspicious_correlation']:
            factors.append({
                'factor': 'Unusual color channel correlation',
                'impact': 0.10,
                'direction': 'ai',
                'severity': 'low'
            })
            ai_score += 0.10
        
        if analysis['color']['color_banding']:
            factors.append({
                'factor': 'Color banding detected (AI artifact)',
                'impact': 0.12,
                'direction': 'ai',
                'severity': 'medium'
            })
            ai_score += 0.12
        elif analysis['color']['entropy'] > 7.0:
            factors.append({
                'factor': f"High color entropy (value: {analysis['color']['entropy']:.2f} bits)",
                'impact': 0.18,
                'direction': 'real',
                'severity': 'medium'
            })
            real_score += 0.18
        
        # AI artifacts detection
        if analysis['artifacts']['unnatural_smoothness']:
            factors.append({
                'factor': f"Unnatural smoothness (Laplacian variance: {analysis['artifacts']['smoothness_score']:.2f})",
                'impact': 0.18,
                'direction': 'ai',
                'severity': 'high'
            })
            ai_score += 0.18
        
        if analysis['artifacts']['has_grid_pattern']:
            factors.append({
                'factor': 'Grid pattern artifacts (diffusion model signature)',
                'impact': 0.22,
                'direction': 'ai',
                'severity': 'high'
            })
            ai_score += 0.22
        
        # Edge density - natural photos have complex edges
        if analysis['compression']['edge_density'] > 0.12:
            factors.append({
                'factor': f"Complex edge patterns (density: {analysis['compression']['edge_density']:.3f})",
                'impact': 0.12,
                'direction': 'real',
                'severity': 'medium'
            })
            real_score += 0.12
        
        return factors, ai_score, real_score

    def _get_enhanced_reasoning(self, prediction, confidence, analysis, factors, ai_score, real_score):
        """Generate comprehensive reasoning"""
        reasons = {
            'summary': [],
            'technical': [],
            'confidence_breakdown': {},
            'risk_assessment': ''
        }
        
        # Confidence assessment
        if confidence < 0.60:
            reasons['summary'].append("⚠️ Low confidence - Image has mixed or ambiguous characteristics")
            reasons['risk_assessment'] = 'inconclusive'
        elif confidence < 0.75:
            reasons['summary'].append("⚠️ Moderate confidence - Some conflicting indicators detected")
            reasons['risk_assessment'] = 'uncertain'
        elif confidence < 0.88:
            reasons['summary'].append("✓ Good confidence level - Multiple consistent indicators")
            reasons['risk_assessment'] = 'likely'
        else:
            reasons['summary'].append("✓ High confidence - Strong evidence from multiple sources")
            reasons['risk_assessment'] = 'confident'
        
        # Add top factors
        sorted_factors = sorted(factors, key=lambda x: x['impact'], reverse=True)
        for factor in sorted_factors[:6]:
            icon = "🔴" if factor['severity'] == 'high' else ("🟡" if factor['severity'] == 'medium' else "🔵")
            reasons['summary'].append(f"{icon} {factor['factor']}")
        
        # Technical details
        reasons['technical'] = [
            f"Noise Level: {analysis['noise']['noise_level']:.2f} ({'Natural range' if analysis['noise']['has_natural_noise'] else 'Suspicious'})",
            f"Texture Consistency: {analysis['texture']['texture_consistency']:.2f}",
            f"Color Entropy: {analysis['color']['entropy']:.2f} bits",
            f"Frequency Balance: {analysis['frequency']['freq_balance']:.3f}",
            f"Color Diversity: {analysis['color']['color_diversity']:.3f}",
            f"Smoothness Score: {analysis['artifacts']['smoothness_score']:.2f}",
            f"EXIF Data: {'Present' if analysis['metadata']['has_exif'] else 'Missing'}",
            f"Compression: {'Natural JPEG' if analysis['compression']['natural_compression'] else 'Unusual/None'}",
            f"Edge Density: {analysis['compression']['edge_density']:.3f}",
            f"Noise Energy: {analysis['noise']['noise_energy']:.0f}"
        ]
        
        # Confidence breakdown
        reasons['confidence_breakdown'] = {
            'ai_indicators_score': round(ai_score, 3),
            'real_indicators_score': round(real_score, 3),
            'model_confidence': round(confidence, 4),
            'total_factors': len(factors),
            'score_difference': round(abs(ai_score - real_score), 3)
        }
        
        return reasons

    def classify_image(self, image, file_size=0):
        try:
            # Comprehensive analysis
            analysis = self._comprehensive_analysis(image)
            analysis['file_size_kb'] = file_size / 1024
            
            # Model prediction
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            confidence_value = float(confidence.item())
            prediction = self.classes[predicted]
            
            # Calculate enhanced confidence factors
            factors, ai_score, real_score = self._calculate_confidence_factors(analysis)
            
            # IMPROVED confidence adjustment logic
            adjusted_confidence = confidence_value
            score_diff = abs(ai_score - real_score)
            
            # If heuristics strongly agree with model
            if (ai_score > real_score and prediction == 'AI-generated') or \
               (real_score > ai_score and prediction == 'Real'):
                # Only boost if score difference is significant
                if score_diff > 0.20:
                    adjusted_confidence = min(0.96, confidence_value + (score_diff * 0.25))
                elif score_diff > 0.10:
                    adjusted_confidence = min(0.92, confidence_value + (score_diff * 0.15))
                else:
                    adjusted_confidence = confidence_value
            else:
                # If heuristics disagree with model
                if score_diff > 0.25:
                    adjusted_confidence = max(0.35, confidence_value - (score_diff * 0.45))
                elif score_diff > 0.15:
                    adjusted_confidence = max(0.45, confidence_value - (score_diff * 0.30))
                else:
                    adjusted_confidence = max(0.50, confidence_value - (score_diff * 0.15))
            
            # CRITICAL OVERRIDE: If real indicators are much stronger, override AI prediction
            if real_score > ai_score + 0.30 and prediction == 'AI-generated':
                prediction = 'Real'
                adjusted_confidence = min(0.85, 0.55 + (real_score - ai_score))
                factors.insert(0, {
                    'factor': 'Prediction overridden: Strong real photo indicators detected',
                    'impact': real_score - ai_score,
                    'direction': 'real',
                    'severity': 'high'
                })
            
            # CRITICAL OVERRIDE: If AI indicators are much stronger, override Real prediction
            elif ai_score > real_score + 0.30 and prediction == 'Real':
                prediction = 'AI-generated'
                adjusted_confidence = min(0.85, 0.55 + (ai_score - real_score))
                factors.insert(0, {
                    'factor': 'Prediction overridden: Strong AI generation indicators detected',
                    'impact': ai_score - real_score,
                    'direction': 'ai',
                    'severity': 'high'
                })
            
            # Get enhanced reasoning
            reasoning = self._get_enhanced_reasoning(
                prediction, adjusted_confidence, analysis, factors, ai_score, real_score
            )
            
            return {
                'prediction': prediction,
                'confidence': round(adjusted_confidence, 4),
                'raw_confidence': round(confidence_value, 4),
                'probabilities': {
                    'AI-generated': round(float(probabilities[0]), 4),
                    'Real': round(float(probabilities[1]), 4)
                },
                'analysis': analysis,
                'reasoning': reasoning,
                'confidence_factors': factors,
                'heuristic_scores': {
                    'ai_score': round(ai_score, 3),
                    'real_score': round(real_score, 3),
                    'agreement': 'strong' if score_diff < 0.10 else ('moderate' if score_diff < 0.20 else 'weak')
                }
            }
            
        except Exception as e:
            print(f"Error in classify_image: {str(e)}")
            raise


@trueshot_ai.route('/')
def index():
    return render_template('trueshot.html')

@trueshot_ai.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
        if not file.filename.lower().endswith(allowed_extensions):
            return jsonify({
                'status': 'error',
                'message': 'Invalid file type. Supported: PNG, JPG, JPEG, WebP, BMP, TIFF'
            }), 400
        
        # Read image
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({
                'status': 'error',
                'message': 'Empty file uploaded'
            }), 400
        
        file_size = len(image_bytes)
        
        # Validate file size (50MB limit)
        if file_size > 50 * 1024 * 1024:
            return jsonify({
                'status': 'error',
                'message': 'File size exceeds 50MB limit'
            }), 400
        
        # Open and process image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode in ('RGBA', 'P', 'L'):
            image = image.convert('RGB')
        
        # Analyze image
        analyzer = AdvancedImageAnalyzer()
        result = analyzer.classify_image(image, file_size)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['image_hash'] = hashlib.md5(image_bytes).hexdigest()[:16]
        result['filename'] = file.filename
        
        # Convert all values to JSON-serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        result = convert_to_serializable(result)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500

@trueshot_ai.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@trueshot_ai.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple images at once"""
    try:
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No images provided'
            }), 400
        
        if len(files) > 10:
            return jsonify({
                'status': 'error',
                'message': 'Maximum 10 images per batch'
            }), 400
        
        analyzer = AdvancedImageAnalyzer()
        results = []
        
        for file in files:
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode in ('RGBA', 'P', 'L'):
                    image = image.convert('RGB')
                
                result = analyzer.classify_image(image, len(image_bytes))
                result['filename'] = file.filename
                
                # Convert to serializable
                def convert_to_serializable(obj):
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_to_serializable(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    return obj
                
                result = convert_to_serializable(result)
                
                results.append({
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                results.append({
                    'status': 'error',
                    'filename': file.filename,
                    'message': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total': len(files),
            'successful': sum(1 for r in results if r['status'] == 'success')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(trueshot_ai)
    app.run(debug=True, port=5000)