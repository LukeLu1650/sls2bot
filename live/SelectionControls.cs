using Godot;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

namespace NecroPilot;

internal static class SelectionControls
{
    internal static (bool IsOpen, NConfirmButton? Button) Preview(NCardGridSelectionScreen screen)
    {
        return screen switch
        {
            NDeckUpgradeSelectScreen => FindPreview(screen, "Confirm", "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer"),
            NDeckEnchantSelectScreen => FindPreview(screen, "Confirm", "%EnchantSinglePreviewContainer", "%EnchantMultiPreviewContainer"),
            NDeckTransformSelectScreen => FindPreview(screen, "Confirm", "%PreviewContainer"),
            NDeckCardSelectScreen => FindPreview(screen, "%PreviewConfirm", "%PreviewContainer"),
            _ => (false, null),
        };
    }

    internal static NConfirmButton? GridConfirm(NCardGridSelectionScreen screen)
    {
        string path = screen is NDeckEnchantSelectScreen or NDeckTransformSelectScreen ? "Confirm" : "%Confirm";
        return Ready(screen.GetNodeOrNull<NConfirmButton>(path));
    }

    private static (bool IsOpen, NConfirmButton? Button) FindPreview(
        Node screen, string buttonPath, string containerPath, string? secondContainerPath = null)
    {
        var container = screen.GetNodeOrNull<Control>(containerPath);
        if (container?.IsVisibleInTree() != true && secondContainerPath != null)
            container = screen.GetNodeOrNull<Control>(secondContainerPath);
        if (container?.IsVisibleInTree() != true)
            return (false, null);
        return (true, Ready(container.GetNodeOrNull<NConfirmButton>(buttonPath)));
    }

    private static NConfirmButton? Ready(NConfirmButton? button) =>
        button is { IsEnabled: true } && button.IsVisibleInTree() ? button : null;
}
