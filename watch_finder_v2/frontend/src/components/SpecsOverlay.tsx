import React, { useEffect } from 'react';
import { X, ExternalLink } from 'lucide-react';
import { Watch, formatPrice, getWatchPrice } from '../data/watchData';

interface SpecsOverlayProps {
  watch: Watch;
  onClose: () => void;
}

const SpecsOverlay: React.FC<SpecsOverlayProps> = ({ watch, onClose }) => {
  const price = getWatchPrice(watch);
  const displayPrice = formatPrice(price);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const specs = [
    { label: 'Movement', value: watch.specs?.movement || watch.movement },
    { label: 'Diameter', value: watch.specs?.diameter_mm ? `${watch.specs.diameter_mm}mm` : watch.case_diameter },
    { label: 'Case Material', value: watch.specs?.case_material || watch.case_material },
    { label: 'Water Resistance', value: watch.specs?.waterproofing_meters ? `${watch.specs.waterproofing_meters}m` : watch.water_resistance },
    { label: 'Crystal', value: watch.specs?.crystal_material },
    { label: 'Power Reserve', value: watch.specs?.power_reserve_hour ? `${watch.specs.power_reserve_hour}h` : undefined },
    { label: 'Dial Color', value: watch.specs?.dial_color },
    { label: 'Strap/Bracelet', value: watch.specs?.strap_1_material || watch.specs?.bracelet_type },
    { label: 'Launch Date', value: watch.specs?.launch_date },
    { label: 'Reference', value: watch.specs?.reference || watch.reference },
  ].filter(spec => spec.value && spec.value !== '-' && spec.value !== 'N/A');

  const complications = [
    { label: 'Date', value: watch.specs?.complication_date },
    { label: 'Chronograph', value: watch.specs?.complication_chronograph },
    { label: 'GMT', value: watch.specs?.complication_gmt },
    { label: 'Dual Time', value: watch.specs?.complication_dual_time },
    { label: 'Power Reserve', value: watch.specs?.complication_power_reserve },
    { label: 'Moonphase', value: watch.specs?.complication_moonphase },
    { label: 'World Time', value: watch.specs?.complication_world_time_zone },
    { label: 'Alarm', value: watch.specs?.complication_alarm },
  ].filter(comp => comp.value && comp.value !== '-' && comp.value !== 'No');

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-in fade-in duration-200"
      onClick={handleBackdropClick}
    >
      <div className="bg-card rounded-xl max-w-2xl w-full max-h-[90vh] overflow-hidden shadow-xl border border-border animate-in slide-in-from-bottom-4 duration-300">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
            <div>
            <h2 className="text-2xl font-semibold text-foreground">{watch.brand}</h2>
            <p className="text-lg text-muted-foreground">{watch.model}</p>
            </div>
          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-muted hover:bg-muted/80 flex items-center justify-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Price */}
          {displayPrice && displayPrice !== 'Contact for price' && (
            <div className="mb-6 p-4 bg-muted/30 rounded-lg">
              <p className="text-sm text-muted-foreground mb-1">Price</p>
              <p className="text-3xl font-bold text-foreground">{displayPrice}</p>
            </div>
          )}

          {/* Algorithm Info */}
          {watch.algorithm && (
            <div className="mb-6 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <p className="text-sm font-medium text-primary mb-1">Recommendation Algorithm</p>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-foreground">{watch.algorithm}</span>
                {watch.confidence && (
                  <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium dark:bg-green-900/30 dark:text-green-400">
                    {watch.confidence.toFixed(1)} confidence
                  </span>
                )}
              </div>
                </div>
              )}

          {/* Main Specifications */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Specifications</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {specs.map((spec) => (
                <div key={spec.label} className="flex justify-between py-2 border-b border-border last:border-0">
                  <span className="text-sm text-muted-foreground">{spec.label}</span>
                  <span className="text-sm font-medium text-foreground text-right">{spec.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Complications */}
          {complications.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Complications</h3>
              <div className="flex flex-wrap gap-2">
                {complications.map((comp) => (
                  <span
                    key={comp.label}
                    className="px-3 py-1 bg-accent text-accent-foreground rounded-full text-sm"
                  >
                    {comp.label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Additional Info */}
          {watch.specs?.specific_info_from_brand && watch.specs.specific_info_from_brand !== '-' && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-foreground mb-2">Additional Information</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {watch.specs.specific_info_from_brand}
              </p>
            </div>
          )}

          {/* Links */}
          <div className="flex flex-col gap-2">
            {watch.product_url && (
              <a
                href={watch.product_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View Product Details
              </a>
            )}
            
          {watch.specs?.brand_website && watch.specs.brand_website !== '-' && (
              <a
                href={watch.specs.brand_website}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Visit Brand Website
              </a>
            )}
            </div>
        </div>
      </div>
    </div>
  );
};

export default SpecsOverlay;
